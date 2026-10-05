import asyncio

from src.workflow import LangGraphMovieAssistant


def test_workflow_routes_preference_request_to_preference_recommender():
    class CoreAssistant:
        async def answer_preference(self, query, preferences):
            return {
                "answer": "Preference results",
                "route": "recommendation",
                "intent": "recommendation",
                "recommendation_mode": "preference_discovery",
                "tools_used": ["discover_movies"],
                "sources": ["TMDB Discover candidates"],
                "query": query,
                "preferences": preferences,
            }

    assistant = LangGraphMovieAssistant(CoreAssistant())

    result = asyncio.run(
        assistant.answer("Recommend science fiction movies from the 1990s")
    )

    assert result["route"] == "recommendation"
    assert result["intent"] == "recommendation"
    assert result["preferences"]["genre_ids"] == [878]
    assert result["preferences"]["year_min"] == 1990
    assert result["preferences"]["year_max"] == 1999
    assert result["workflow_steps"] == [
        "understand_request",
        "run_preference_request",
        "generate_explanation",
        "validate_explanation",
    ]


def test_workflow_routes_give_me_movies_phrase_to_preference_recommender():
    class CoreAssistant:
        async def answer_preference(self, query, preferences):
            return {
                "answer": "Preference results",
                "route": "recommendation",
                "intent": "recommendation",
                "tools_used": ["discover_movies"],
                "sources": [],
                "preferences": preferences,
            }

    query = "Give me all movies that has a bit of comedy, romance and good feel to it"
    result = asyncio.run(LangGraphMovieAssistant(CoreAssistant()).answer(query))

    assert result["intent"] == "recommendation"
    assert set(result["preferences"]["genre_ids"]) == {35, 10749}
    assert "run_preference_request" in result["workflow_steps"]


def test_workflow_uses_explicit_trending_gemma_and_validator_nodes():
    class CoreAssistant:
        async def prepare_for_intent(self, query, intent, region):
            assert intent.value == "trending"
            return {
                "answer": "Trending fallback",
                "route": "mcp",
                "intent": intent.value,
                "tools_used": ["get_trending_movies"],
                "sources": ["TMDB"],
                "_explanation_context": {
                    "kind": "grounded",
                    "verified_answer": "Trending fallback",
                },
            }

        def generate_explanation(self, query, context):
            return context["verified_answer"] + "\nA grounded explanation."

        def validate_explanation(self, answer, context):
            assert context["verified_answer"] in answer

    result = asyncio.run(
        LangGraphMovieAssistant(CoreAssistant()).answer("What is trending this week?")
    )

    assert result["answer"].endswith("A grounded explanation.")
    assert result["workflow_steps"] == [
        "understand_request",
        "run_trending",
        "generate_explanation",
        "validate_explanation",
    ]
    assert "Fine-tuned Gemma response synthesis" in result["sources"]


def test_workflow_uses_deterministic_fallback_after_failed_validation():
    class CoreAssistant:
        async def prepare_for_intent(self, query, intent, region):
            return {
                "answer": "Verified TMDB answer",
                "route": "mcp",
                "intent": intent.value,
                "tools_used": ["get_movie_details"],
                "sources": ["TMDB"],
            }

        def generate_explanation(self, query, context):
            return "Hallucinated answer"

        def validate_explanation(self, answer, context):
            raise ValueError("verified response was omitted")

    result = asyncio.run(
        LangGraphMovieAssistant(CoreAssistant()).answer("Who directed Arrival?")
    )

    assert result["answer"] == "Verified TMDB answer"
    assert result["workflow_steps"][-2:] == [
        "validate_explanation",
        "deterministic_fallback",
    ]
    assert "Deterministic response formatting" in result["sources"]


def test_workflow_skips_synthesis_for_empty_tmdb_result():
    class CoreAssistant:
        async def prepare_for_intent(self, query, intent, region):
            return {
                "answer": "I couldn't find any TMDB movies matching that query.",
                "route": "mcp",
                "intent": intent.value,
                "tools_used": ["search_movies"],
                "sources": ["TMDB"],
                "_explanation_context": {
                    "kind": "grounded",
                    "verified_answer": (
                        "I couldn't find any TMDB movies matching that query."
                    ),
                    "allow_synthesis": False,
                },
            }

        def generate_explanation(self, query, context):
            raise AssertionError("error responses must not be synthesized")

    result = asyncio.run(
        LangGraphMovieAssistant(CoreAssistant()).answer("Unknown movie")
    )

    assert result["answer"] == "I couldn't find any TMDB movies matching that query."
    assert "Fine-tuned Gemma response synthesis" not in result["sources"]


def test_workflow_general_question_uses_mcp_route():
    class CoreAssistant:
        async def prepare_for_intent(self, query, intent, region):
            assert query == "Explain film noir"
            assert intent.value == "general"
            return {
                "answer": "TMDB movie search results: Film Noir",
                "route": "mcp",
                "intent": intent.value,
                "tools_used": ["search_movies"],
                "sources": ["TMDB"],
            }

    result = asyncio.run(
        LangGraphMovieAssistant(CoreAssistant()).answer("Explain film noir")
    )

    assert result["route"] == "mcp"
    assert result["intent"] == "general"
    assert result["tools_used"] == ["search_movies"]
    assert result["sources"] == ["TMDB"]
    assert result["workflow_steps"] == [
        "understand_request",
        "run_general_question",
        "generate_explanation",
        "validate_explanation",
    ]
