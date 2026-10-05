import asyncio
from unittest.mock import AsyncMock

from src.hybrid_assistant import HybridMovieAssistant
from src.recommender import MovieRecommender
from src.router import Intent


class FakeEmbeddings:
    def embed_query(self, _text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_general_question_uses_mcp_movie_search():
    assistant = HybridMovieAssistant()
    assistant.mcp.call_tool = AsyncMock(
        return_value=[
            {
                "id": 1,
                "title": "Film Noir",
                "release_year": "2007",
                "overview": "A documentary about film noir.",
            }
        ]
    )

    result = asyncio.run(
        assistant.answer_for_intent("Explain film noir", Intent.GENERAL)
    )

    assert result["route"] == "mcp"
    assert result["intent"] == "general"
    assert result["tools_used"] == ["search_movies"]
    assert result["sources"] == ["TMDB"]
    assert "Film Noir (2007)" in result["answer"]
    assistant.mcp.call_tool.assert_awaited_once_with(
        "search_movies",
        {"query": "film noir", "limit": 5},
    )


def test_plot_question_uses_movie_details_tool():
    assistant = HybridMovieAssistant()
    assistant.mcp.call_tool = AsyncMock(
        return_value={
            "id": 27205,
            "title": "Inception",
            "release_year": 2010,
            "overview": "A thief enters dreams to steal and plant ideas.",
            "cast": [],
            "genres": ["Science Fiction"],
        }
    )

    result = asyncio.run(
        assistant.answer_for_intent(
            "/Whats the plot of Inception", Intent.MOVIE_DETAILS
        )
    )

    assert "A thief enters dreams" in result["answer"]
    assistant.mcp.call_tool.assert_awaited_once_with(
        "get_movie_details",
        {"title": "Inception"},
    )


def test_ending_question_does_not_present_overview_as_the_ending():
    assistant = HybridMovieAssistant()
    assistant.mcp.call_tool = AsyncMock(
        return_value={
            "id": 27205,
            "title": "Inception",
            "release_year": 2010,
            "overview": "A thief enters dreams to steal and plant ideas.",
        }
    )

    result = asyncio.run(
        assistant.answer_for_intent(
            "What's the ending of Inception?", Intent.MOVIE_ENDING
        )
    )

    assert result["intent"] == "movie_ending"
    assert "does not provide a full ending explanation" in result["answer"]
    assert "TMDB overview:" in result["answer"]
    assistant.mcp.call_tool.assert_awaited_once_with(
        "get_movie_details",
        {"title": "Inception"},
    )


def test_reranked_response_uses_recommendation_route(monkeypatch):
    assistant = HybridMovieAssistant(recommender=MovieRecommender(FakeEmbeddings()))
    assistant.mcp.call_tool = AsyncMock(
        return_value={
            "source_movie": {
                "id": 1,
                "title": "Interstellar",
                "overview": "Space exploration",
                "genre_ids": [878],
            },
            "recommendations": [
                {
                    "id": 2,
                    "title": "Ad Astra",
                    "overview": "Space journey",
                    "genre_ids": [878],
                }
            ],
        }
    )
    monkeypatch.setattr("src.hybrid_assistant.classify_intent", lambda _query: Intent.SIMILAR_MOVIES)
    monkeypatch.setattr("src.hybrid_assistant.extract_title", lambda _query, _intent: "Interstellar")
    monkeypatch.setattr("src.hybrid_assistant.extract_year", lambda _query: None)

    result = asyncio.run(assistant.answer("Movies like Interstellar"))

    assert result["route"] == "recommendation"
    assert "Compared with Interstellar" in result["answer"]
    assert "Science Fiction" in result["answer"]
    assert "Signals:" not in result["answer"]


def test_recommendation_route_diversifies_a_larger_relevance_pool(monkeypatch):
    class RecordingRecommender:
        def __init__(self):
            self.limit = None

        def rank(self, _source, candidates, limit):
            self.limit = limit
            return [
                {**movie, "score": 1 - index / 10, "reason": "Ranked"}
                for index, movie in enumerate(candidates[:limit])
            ]

    class RecordingDiversityReranker:
        def __init__(self):
            self.pool_size = None
            self.limit = None

        def rerank(self, candidates, limit):
            self.pool_size = len(candidates)
            self.limit = limit
            return list(reversed(candidates))[:limit]

    recommender = RecordingRecommender()
    diversity = RecordingDiversityReranker()
    assistant = HybridMovieAssistant(
        recommender=recommender,
        diversity_reranker=diversity,
        recommendation_limit=2,
        diversity_candidate_pool=3,
    )
    assistant.mcp.call_tool = AsyncMock(
        return_value={
            "source_movie": {"id": 1, "title": "Source"},
            "recommendations": [
                {"id": 2, "title": "First"},
                {"id": 3, "title": "Second"},
                {"id": 4, "title": "Third"},
                {"id": 5, "title": "Fourth"},
            ],
        }
    )
    monkeypatch.setattr(
        "src.hybrid_assistant.classify_intent",
        lambda _query: Intent.SIMILAR_MOVIES,
    )
    monkeypatch.setattr(
        "src.hybrid_assistant.extract_title",
        lambda _query, _intent: "Source",
    )
    monkeypatch.setattr("src.hybrid_assistant.extract_year", lambda _query: None)

    result = asyncio.run(assistant.answer("Movies like Source"))

    assert recommender.limit == 3
    assert diversity.pool_size == 3
    assert diversity.limit == 2
    assert "Third" in result["answer"]
    assert "Second" in result["answer"]
    assert "First" not in result["answer"]


def test_recommendation_route_synthesizes_final_ranked_movies(monkeypatch):
    class RecordingSynthesizer:
        def __init__(self):
            self.recommendations = None

        def generate(self, *, recommendations, **_kwargs):
            self.recommendations = recommendations
            return "Synthesized recommendation"

    synthesizer = RecordingSynthesizer()
    assistant = HybridMovieAssistant(
        recommender=MovieRecommender(FakeEmbeddings()),
        response_synthesizer=synthesizer,
    )
    assistant.mcp.call_tool = AsyncMock(
        return_value={
            "source_movie": {
                "id": 1,
                "title": "Interstellar",
                "overview": "Space exploration",
                "genre_ids": [878],
            },
            "recommendations": [
                {
                    "id": 2,
                    "title": "Ad Astra",
                    "overview": "Space journey",
                    "genre_ids": [878],
                }
            ],
        }
    )
    monkeypatch.setattr(
        "src.hybrid_assistant.extract_title",
        lambda _query, _intent: "Interstellar",
    )
    monkeypatch.setattr("src.hybrid_assistant.extract_year", lambda _query: None)

    result = asyncio.run(
        assistant.answer_for_intent("Movies like Interstellar", Intent.SIMILAR_MOVIES)
    )

    assert result["answer"] == "Synthesized recommendation"
    assert synthesizer.recommendations[0]["title"] == "Ad Astra"
    assert synthesizer.recommendations[0]["score"] is not None
    assert "Fine-tuned Gemma response synthesis" in result["sources"]


def test_unranked_tmdb_response_keeps_mcp_route(monkeypatch):
    assistant = HybridMovieAssistant()
    assistant.mcp.call_tool = AsyncMock(return_value=[])
    monkeypatch.setattr("src.hybrid_assistant.classify_intent", lambda _query: Intent.TRENDING)

    result = asyncio.run(assistant.answer("What is trending?"))

    assert result["route"] == "mcp"


def test_answer_for_intent_uses_langgraph_selected_route(monkeypatch):
    assistant = HybridMovieAssistant()
    assistant.mcp.call_tool = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "src.hybrid_assistant.classify_intent",
        lambda _query: (_ for _ in ()).throw(AssertionError("must not reroute")),
    )

    result = asyncio.run(
        assistant.answer_for_intent("What is trending?", Intent.TRENDING)
    )

    assert result["route"] == "mcp"
    assistant.mcp.call_tool.assert_awaited_once_with(
        "get_trending_movies",
        {"time_window": "week", "limit": 5},
    )


def test_preference_discovery_handles_mcp_error_payload():
    assistant = HybridMovieAssistant(preference_recommender=object())
    assistant.mcp.call_tool = AsyncMock(return_value={"error": "TMDB unavailable"})

    result = asyncio.run(
        assistant.answer_preference(
            "Recommend science fiction movies",
            {
                "query": "Recommend science fiction movies",
                "genre_ids": [878],
                "genres": ["Science Fiction"],
                "year_min": None,
                "year_max": None,
            },
        )
    )

    assert result["route"] == "recommendation_error"


def test_preference_route_synthesizes_only_after_final_reranking():
    events = []

    class PreferenceRecommender:
        def rank(self, _preferences, candidates, _limit):
            events.append("rank")
            return list(candidates)

    class DiversityReranker:
        def rerank(self, candidates, limit):
            events.append("rerank")
            return list(reversed(candidates))[:limit]

    class Synthesizer:
        def __init__(self):
            self.recommendations = None

        def generate(self, *, recommendations, **_kwargs):
            events.append("synthesize")
            self.recommendations = recommendations
            return "Final explanation"

    synthesizer = Synthesizer()
    assistant = HybridMovieAssistant(
        preference_recommender=PreferenceRecommender(),
        diversity_reranker=DiversityReranker(),
        response_synthesizer=synthesizer,
        recommendation_limit=2,
    )
    assistant.mcp.call_tool = AsyncMock(
        return_value=[
            {"id": 1, "title": "First"},
            {"id": 2, "title": "Second"},
            {"id": 3, "title": "Third"},
        ]
    )

    result = asyncio.run(
        assistant.answer_preference(
            "Recommend science fiction movies",
            {"genres": ["Science Fiction"], "genre_ids": [878]},
        )
    )

    assert events == ["rank", "rerank", "synthesize"]
    assert [movie["title"] for movie in synthesizer.recommendations] == [
        "Third",
        "Second",
    ]
    assert result["answer"] == "Final explanation"
    assert "Fine-tuned Gemma response synthesis" in result["sources"]
