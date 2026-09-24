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
    assert "Signals: semantic plot similarity, genre overlap" in result["answer"]


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


def test_unranked_tmdb_response_keeps_mcp_route(monkeypatch):
    assistant = HybridMovieAssistant()
    assistant.mcp.call_tool = AsyncMock(return_value=[])
    monkeypatch.setattr("src.hybrid_assistant.classify_intent", lambda _query: Intent.TRENDING)

    result = asyncio.run(assistant.answer("What is trending?"))

    assert result["route"] == "mcp"
