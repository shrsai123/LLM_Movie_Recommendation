import asyncio
import json
import logging

from src.candidate_merger import merge_candidates
from src.mcp_client import MovieMCPClient, MovieMCPError
from src.router import Intent, classify_intent, extract_title, extract_year

logger = logging.getLogger(__name__)


class HybridMovieAssistant:
    def __init__(
        self,
        qa_chain=None,
        recommender=None,
        diversity_reranker=None,
        candidate_retriever=None,
        tmdb_candidate_limit: int = 20,
        faiss_candidate_limit: int = 20,
        recommendation_limit: int = 5,
        diversity_candidate_pool: int = 15,
    ):
        if diversity_candidate_pool < 1:
            raise ValueError("diversity_candidate_pool must be positive")
        self.qa_chain = qa_chain
        self.mcp = MovieMCPClient()
        self.recommender = recommender
        self.diversity_reranker = diversity_reranker
        self.candidate_retriever = candidate_retriever
        self.tmdb_candidate_limit = tmdb_candidate_limit
        self.faiss_candidate_limit = faiss_candidate_limit
        self.recommendation_limit = recommendation_limit
        self.diversity_candidate_pool = diversity_candidate_pool

    async def answer(self, query: str, region: str = "US") -> dict:
        intent = classify_intent(query)
        ranked_recommendations = False
        faiss_candidates_used = False

        if intent == Intent.RAG:
            result = await asyncio.to_thread(
                self.qa_chain.invoke,
                {"query": query},
            )

            return {
                "answer": result["result"],
                "route": "rag",
                "intent": intent.value,
                "tools_used": [],
                "sources": ["TMDB 5000 FAISS index"],
            }

        if intent == Intent.TRENDING:
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

        return {
            "answer": self._format_result(tool, data),
            "route": "recommendation" if ranked_recommendations else "mcp",
            "intent": intent.value,
            "tools_used": [tool],
            "sources": (
                [
                    "TMDB candidates and metadata",
                    "FAISS movie candidates",
                    "Local four-signal ranking",
                ]
                if faiss_candidates_used
                else ["TMDB candidates and metadata", "Local four-signal ranking"]
            )
            if ranked_recommendations
            else ["TMDB"],
        }

    @staticmethod
    def _format_result(tool: str, data: dict | list) -> str:
        if isinstance(data, dict) and data.get("error"):
            return data["error"]

        if tool == "get_trending_movies":
            lines = [
                f"- {movie['title']} ({movie.get('release_year', 'Unknown')})" for movie in data
            ]
            return "Trending movies this week:\n" + "\n".join(lines)

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

            lines = [
                f"- {HybridMovieAssistant._movie_label(movie)}"
                + (f" — {movie['reason']}" if movie.get("reason") else "")
                for movie in recommendations
            ]
            return f"Movies similar to {source}:\n" + "\n".join(lines)

        return "I got a response, but I do not know how to format it yet."

    @staticmethod
    def _movie_label(movie: dict) -> str:
        title = movie.get("title", "Unknown title")
        year = movie.get("release_year")
        return f"{title} ({year})" if year else title
