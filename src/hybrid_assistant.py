import asyncio
import json
import logging

from src.candidate_merger import merge_candidates
from src.mcp_client import MovieMCPClient, MovieMCPError
from src.preferences import extract_preferences
from src.router import (
    Intent,
    classify_intent,
    extract_search_query,
    extract_title,
    extract_year,
)

logger = logging.getLogger(__name__)


class HybridMovieAssistant:
    def __init__(
        self,
        recommender=None,
        preference_recommender=None,
        diversity_reranker=None,
        candidate_retriever=None,
        response_synthesizer=None,
        tmdb_candidate_limit: int = 20,
        faiss_candidate_limit: int = 20,
        recommendation_limit: int = 5,
        diversity_candidate_pool: int = 15,
    ):
        if diversity_candidate_pool < 1:
            raise ValueError("diversity_candidate_pool must be positive")
        self.mcp = MovieMCPClient()
        self.recommender = recommender
        self.preference_recommender = preference_recommender
        self.diversity_reranker = diversity_reranker
        self.candidate_retriever = candidate_retriever
        self.tmdb_candidate_limit = tmdb_candidate_limit
        self.faiss_candidate_limit = faiss_candidate_limit
        self.recommendation_limit = recommendation_limit
        self.diversity_candidate_pool = diversity_candidate_pool
        self.response_synthesizer = response_synthesizer

    async def answer(self, query: str, region: str = "US") -> dict:
        """Compatibility entry point for callers that do not use LangGraph."""
        intent = classify_intent(query)
        if intent == Intent.RECOMMENDATION:
            return await self.answer_preference(query, extract_preferences(query))
        return await self.answer_for_intent(query, intent, region)

    async def answer_for_intent(
        self,
        query: str,
        intent: Intent,
        region: str = "US",
        synthesize_response: bool = True,
    ) -> dict:
        """Execute an intent selected by the LangGraph workflow."""
        ranked_recommendations = False
        faiss_candidates_used = False
        synthesized_answer = None
        synthesis_used = False
        explanation_context = None

        if intent == Intent.GENERAL:
            tool = "search_movies"
            arguments = {"query": extract_search_query(query), "limit": 5}
            year = extract_year(query)
            if year:
                arguments["year"] = year
            try:
                data = await self.mcp.call_tool(tool, arguments)
            except MovieMCPError as exc:
                return {
                    "answer": f"I couldn't search TMDB right now. {exc}",
                    "route": "mcp_error",
                    "intent": intent.value,
                    "tools_used": [tool],
                    "sources": ["TMDB"],
                }

        elif intent == Intent.TRENDING:
            tool = "get_trending_movies"
            try:
                data = await self.mcp.call_tool(
                    tool,
                    {"time_window": "week", "limit": 5},
                )
            except MovieMCPError as exc:
                return {
                    "answer": f"I couldn't get trending movies from TMDB right now. {exc}",
                    "route": "mcp_error",
                    "intent": intent.value,
                    "tools_used": [tool],
                    "sources": ["TMDB"],
                }

        else:
            title = extract_title(query, intent)

            if not title:
                return {
                    "answer": "Which movie are you referring to?",
                    "route": "clarification",
                    "intent": intent.value,
                    "tools_used": [],
                    "sources": [],
                }

            tool_mapping = {
                Intent.WATCH_PROVIDERS: "get_watch_providers",
                Intent.MOVIE_DETAILS: "get_movie_details",
                Intent.MOVIE_ENDING: "get_movie_details",
                Intent.SIMILAR_MOVIES: "get_similar_movies",
            }

            tool = tool_mapping[intent]
            arguments = {"title": title}
            year = extract_year(query)

            if year:
                arguments["year"] = year

            if intent == Intent.WATCH_PROVIDERS:
                arguments["region"] = region.upper()
            elif intent == Intent.SIMILAR_MOVIES and self.recommender:
                arguments["limit"] = self.tmdb_candidate_limit

            try:
                data = await self.mcp.call_tool(tool, arguments)
            except MovieMCPError as exc:
                return {
                    "answer": f"I couldn't get that from TMDB right now. {exc}",
                    "route": "mcp_error",
                    "intent": intent.value,
                    "tools_used": [tool],
                    "sources": ["TMDB"],
                }

            if intent == Intent.SIMILAR_MOVIES and self.recommender and not data.get("error"):
                tmdb_candidates = data.get("recommendations") or []
                faiss_candidates = []

                if self.candidate_retriever:
                    try:
                        faiss_candidates = await asyncio.to_thread(
                            self.candidate_retriever.search,
                            data["source_movie"],
                            self.faiss_candidate_limit,
                        )
                        faiss_candidates_used = bool(faiss_candidates)
                    except Exception:
                        logger.exception("FAISS candidate retrieval failed; using TMDB only")

                merged_candidates = merge_candidates(
                    ("tmdb", tmdb_candidates),
                    ("faiss", faiss_candidates),
                )

                logger.info(
                    "SOURCE METADATA: title=%s keywords=%s collection_id=%s genres=%s",
                    data["source_movie"].get("title"),
                    data["source_movie"].get("keywords"),
                    data["source_movie"].get("collection_id"),
                    data["source_movie"].get("genre_ids"),
                )

                if faiss_candidates:
                    logger.info(
                        "FAISS SAMPLE METADATA: title=%s keywords=%s collection_id=%s genres=%s",
                        faiss_candidates[0].get("title"),
                        faiss_candidates[0].get("keywords"),
                        faiss_candidates[0].get("collection_id"),
                        faiss_candidates[0].get("genre_ids"),
                    )

                if tmdb_candidates:
                    logger.info(
                        "TMDB SAMPLE METADATA: title=%s keywords=%s collection_id=%s genres=%s",
                        tmdb_candidates[0].get("title"),
                        tmdb_candidates[0].get("keywords"),
                        tmdb_candidates[0].get("collection_id"),
                        tmdb_candidates[0].get("genre_ids"),
                    )

                ranking_limit = (
                    max(self.diversity_candidate_pool, self.recommendation_limit)
                    if self.diversity_reranker
                    else self.recommendation_limit
                )
                relevance_ranked = await asyncio.to_thread(
                    self.recommender.rank,
                    data["source_movie"],
                    merged_candidates,
                    ranking_limit,
                )
                final_recommendations = (
                    self.diversity_reranker.rerank(
                        relevance_ranked,
                        self.recommendation_limit,
                    )
                    if self.diversity_reranker
                    else relevance_ranked
                )
                data["recommendations"] = final_recommendations

                explanation_context = {
                    "kind": "recommendations",
                    "recommendations": final_recommendations,
                    "source_movie": data["source_movie"],
                }

                if synthesize_response and self.response_synthesizer is not None:
                    try:
                        synthesized_answer = await asyncio.to_thread(
                            self.response_synthesizer.generate,
                            query=query,
                            recommendations=final_recommendations,
                            source_movie=data["source_movie"],
                        )
                        synthesis_used = True
                    except Exception:
                        # If the model omits or reorders a movie, generate()
                        # validation fails and the deterministic formatted
                        # answer is returned.
                        logger.exception(
                            "Fine-tuned response synthesis failed; using formatter"
                        )

                duplicates_removed = (
                    len(tmdb_candidates)
                    + len(faiss_candidates)
                    - len(merged_candidates)
                )
                data["candidate_counts"] = {
                    "tmdb": len(tmdb_candidates),
                    "faiss": len(faiss_candidates),
                    "merged": len(merged_candidates),
                    "duplicates_removed": duplicates_removed,
                    "relevance_ranked": len(relevance_ranked),
                    "final": len(final_recommendations),
                }

                def candidate_view(movie: dict) -> dict:
                    return {
                        "id": movie.get("id"),
                        "title": movie.get("title"),
                        "year": movie.get("release_year"),
                        "candidate_sources": movie.get("candidate_sources", []),
                        "score": movie.get("score"),
                        "signals": movie.get("signals"),
                        "relevance_rank": movie.get("relevance_rank"),
                        "diversity_penalty": movie.get("diversity_penalty"),
                        "diversity_score": movie.get("diversity_score"),
                    }

                logger.info(
                    "CANDIDATE COUNTS: %s",
                    json.dumps(data["candidate_counts"], indent=2),
                )
                logger.info(
                    "TMDB CANDIDATES:\n%s",
                    json.dumps(
                        [candidate_view(movie) for movie in tmdb_candidates],
                        indent=2,
                    ),
                )
                logger.info(
                    "FAISS CANDIDATES:\n%s",
                    json.dumps(
                        [candidate_view(movie) for movie in faiss_candidates],
                        indent=2,
                    ),
                )
                logger.info(
                    "MERGED CANDIDATES:\n%s",
                    json.dumps(
                        [candidate_view(movie) for movie in merged_candidates],
                        indent=2,
                    ),
                )
                logger.info(
                    "RELEVANCE-RANKED POOL:\n%s",
                    json.dumps(
                        [candidate_view(movie) for movie in relevance_ranked],
                        indent=2,
                    ),
                )
                logger.info(
                    "FINAL RANKED LIST:\n%s",
                    json.dumps(
                        [candidate_view(movie) for movie in final_recommendations],
                        indent=2,
                    ),
                )
                ranked_recommendations = True

        deterministic_answer = (
            self._format_ending_result(data)
            if intent == Intent.MOVIE_ENDING
            else self._format_result(tool, data)
        )
        if explanation_context is None:
            explanation_context = {
                "kind": "grounded",
                "verified_answer": deterministic_answer,
                # Error and empty-result messages are already complete. Passing
                # them through the small synthesis model can only add noise.
                "allow_synthesis": (
                    intent != Intent.MOVIE_ENDING
                    and not deterministic_answer.casefold().startswith("i couldn't")
                ),
            }

        result = {
            "answer": synthesized_answer or deterministic_answer,
            "route": "recommendation" if ranked_recommendations else "mcp",
            "intent": intent.value,
            "tools_used": [tool],
            "sources": (
                [
                    "TMDB candidates and metadata",
                    *(["FAISS movie candidates"] if faiss_candidates_used else []),
                    "Local four-signal ranking",
                    *(
                        ["Fine-tuned Gemma response synthesis"]
                        if synthesis_used
                        else ["Deterministic response formatting"]
                        if synthesize_response
                        else []
                    ),
                ]
                if ranked_recommendations
                else ["TMDB"]
            ),
        }
        if not synthesize_response:
            result["_explanation_context"] = explanation_context
        return result

    async def answer_preference(
        self,
        query: str,
        preferences: dict,
        synthesize_response: bool = True,
    ) -> dict:
        """Discover and rank movies without requiring a source movie."""
        if self.preference_recommender is None:
            raise RuntimeError("Preference recommender has not been configured")

        tool = "discover_movies"
        arguments = {
            "genre_ids": preferences.get("genre_ids") or None,
            "year_min": preferences.get("year_min"),
            "year_max": preferences.get("year_max"),
            "limit": self.tmdb_candidate_limit,
        }
        arguments = {key: value for key, value in arguments.items() if value is not None}

        try:
            tmdb_candidates = await self.mcp.call_tool(tool, arguments)
        except MovieMCPError as exc:
            tmdb_candidates = []
            logger.warning("TMDB discovery failed; trying FAISS candidates: %s", exc)

        if isinstance(tmdb_candidates, dict):
            logger.warning(
                "TMDB discovery returned an error: %s",
                tmdb_candidates.get("error", "unexpected response payload"),
            )
            tmdb_candidates = []

        faiss_candidates = []
        if self.candidate_retriever:
            try:
                faiss_candidates = await asyncio.to_thread(
                    self.candidate_retriever.search_query,
                    query,
                    self.faiss_candidate_limit,
                )
            except Exception:
                logger.exception("FAISS preference retrieval failed")

        merged_candidates = merge_candidates(
            ("tmdb", tmdb_candidates),
            ("faiss", faiss_candidates),
        )
        if not merged_candidates:
            return {
                "answer": "I couldn't find movies matching those preferences right now.",
                "route": "recommendation_error",
                "intent": "recommendation",
                "recommendation_mode": "preference_discovery",
                "tools_used": [tool],
                "sources": [],
            }
        relevance_limit = (
            max(self.diversity_candidate_pool, self.recommendation_limit)
            if self.diversity_reranker
            else self.recommendation_limit
        )
        relevance_ranked = await asyncio.to_thread(
            self.preference_recommender.rank,
            preferences,
            merged_candidates,
            relevance_limit,
        )
        recommendations = (
            self.diversity_reranker.rerank(relevance_ranked, self.recommendation_limit)
            if self.diversity_reranker
            else relevance_ranked
        )

        synthesized_answer = None
        synthesis_used = False
        if synthesize_response and self.response_synthesizer is not None:
            try:
                # Candidate discovery and both ranking passes are complete. The
                # fine-tuned model only explains this immutable final list.
                synthesized_answer = await asyncio.to_thread(
                    self.response_synthesizer.generate,
                    query=query,
                    recommendations=recommendations,
                    preferences=preferences,
                )
                synthesis_used = True
            except Exception:
                logger.exception(
                    "Fine-tuned response synthesis failed; using formatter"
                )

        deterministic_answer = self._format_preference_result(
            preferences,
            recommendations,
        )
        result = {
            "answer": synthesized_answer or deterministic_answer,
            "route": "recommendation",
            "intent": "recommendation",
            "recommendation_mode": "preference_discovery",
            "tools_used": [tool],
            "sources": [
                *(["TMDB Discover candidates"] if tmdb_candidates else []),
                *(["FAISS movie candidates"] if faiss_candidates else []),
                "Local preference ranking",
                *(["Diversity reranking"] if self.diversity_reranker else []),
                *(
                    ["Fine-tuned Gemma response synthesis"]
                    if synthesis_used
                    else ["Deterministic response formatting"]
                    if synthesize_response
                    else []
                ),
            ],
            "candidate_counts": {
                "tmdb": len(tmdb_candidates),
                "faiss": len(faiss_candidates),
                "merged": len(merged_candidates),
                "diversity_pool": len(relevance_ranked),
            },
        }
        if not synthesize_response:
            result["_explanation_context"] = {
                "kind": "recommendations",
                "recommendations": recommendations,
                "preferences": preferences,
            }
        return result

    async def prepare_for_intent(
        self,
        query: str,
        intent: Intent,
        region: str = "US",
    ) -> dict:
        """Run retrieval/ranking while deferring explanation to LangGraph."""
        return await self.answer_for_intent(
            query,
            intent,
            region,
            synthesize_response=False,
        )

    async def prepare_preference(self, query: str, preferences: dict) -> dict:
        """Run discovery/ranking while deferring explanation to LangGraph."""
        return await self.answer_preference(
            query,
            preferences,
            synthesize_response=False,
        )

    def generate_explanation(self, query: str, context: dict) -> str:
        """Generate the shared post-retrieval explanation for the graph."""
        if self.response_synthesizer is None:
            raise RuntimeError("Fine-tuned Gemma synthesizer is unavailable")

        if context.get("kind") == "recommendations":
            return self.response_synthesizer.generate(
                query=query,
                recommendations=context["recommendations"],
                source_movie=context.get("source_movie"),
                preferences=context.get("preferences"),
            )
        return self.response_synthesizer.generate_grounded(
            query=query,
            verified_answer=context["verified_answer"],
        )

    def validate_explanation(self, answer: str, context: dict) -> None:
        """Validate generated text against the immutable retrieval result."""
        if self.response_synthesizer is None:
            raise RuntimeError("Fine-tuned Gemma synthesizer is unavailable")

        if context.get("kind") == "recommendations":
            self.response_synthesizer.validate(
                answer,
                context["recommendations"],
            )
            return
        self.response_synthesizer.validate_grounded(
            answer,
            context["verified_answer"],
        )

    @staticmethod
    def _format_preference_result(preferences: dict, recommendations: list[dict]) -> str:
        genres = preferences.get("genres") or []
        description = ", ".join(genres) if genres else "your preferences"
        lines = [
            f"- {HybridMovieAssistant._movie_label(movie)}"
            + (f" — {movie['reason']}" if movie.get("reason") else "")
            for movie in recommendations
        ]
        return f"Movies matching {description}:\n" + "\n".join(lines)

    @staticmethod
    def _format_ending_result(data: dict) -> str:
        if data.get("error"):
            return data["error"]

        title = HybridMovieAssistant._movie_label(data)
        response = (
            f"TMDB does not provide a full ending explanation for {title}, so I "
            "can't answer that reliably from the available source."
        )
        if data.get("overview"):
            response += f"\n\nTMDB overview: {data['overview']}"
        return response

    @staticmethod
    def _format_result(tool: str, data: dict | list) -> str:
        if isinstance(data, dict) and data.get("error"):
            return data["error"]

        if tool == "get_trending_movies":
            lines = [
                f"- {movie['title']} ({movie.get('release_year', 'Unknown')})" for movie in data
            ]
            return "Trending movies this week:\n" + "\n".join(lines)

        if tool == "search_movies":
            if not data:
                return "I couldn't find any TMDB movies matching that query."
            lines = [
                f"- {HybridMovieAssistant._movie_label(movie)}"
                + (f" — {movie['overview']}" if movie.get("overview") else "")
                for movie in data
            ]
            return "TMDB movie search results:\n" + "\n".join(lines)

        if tool == "get_watch_providers":
            movie = data["movie"]
            title = HybridMovieAssistant._movie_label(movie)
            sections = []

            for label, key in [
                ("Streaming", "stream"),
                ("Rent", "rent"),
                ("Buy", "buy"),
            ]:
                providers = data.get(key) or []
                if providers:
                    sections.append(f"{label}: {', '.join(providers)}")

            if not sections:
                return f"I couldn't find US streaming, rental, or purchase providers for {title}."

            response = f"Here is where you can watch {title} in {data.get('region', 'US')}:\n"
            response += "\n".join(f"- {section}" for section in sections)

            if data.get("tmdb_link"):
                response += f"\n\nTMDB: {data['tmdb_link']}"

            return response

        if tool == "get_movie_details":
            title = HybridMovieAssistant._movie_label(data)
            cast = ", ".join(data.get("cast") or [])
            genres = ", ".join(data.get("genres") or [])
            lines = [f"{title}"]

            if data.get("director"):
                lines.append(f"Director: {data['director']}")
            if cast:
                lines.append(f"Cast: {cast}")
            if genres:
                lines.append(f"Genres: {genres}")
            if data.get("runtime"):
                lines.append(f"Runtime: {data['runtime']} minutes")
            if data.get("overview"):
                lines.append(f"\n{data['overview']}")

            return "\n".join(lines)

        if tool == "get_similar_movies":
            source = HybridMovieAssistant._movie_label(data["source_movie"])
            recommendations = data.get("recommendations") or []

            if not recommendations:
                return f"I couldn't find TMDB recommendations similar to {source}."

            lines = []
            for rank, movie in enumerate(recommendations, start=1):
                explanation = movie.get("comparison") or movie.get("reason") or ""
                lines.append(
                    f"{rank}. {HybridMovieAssistant._movie_label(movie)}"
                    + (f"\n{explanation}" if explanation else "")
                )
            return f"Movies similar to {source}:\n" + "\n".join(lines)

        return "I got a response, but I do not know how to format it yet."

    @staticmethod
    def _movie_label(movie: dict) -> str:
        title = movie.get("title", "Unknown title")
        year = movie.get("release_year")
        return f"{title} ({year})" if year else title
