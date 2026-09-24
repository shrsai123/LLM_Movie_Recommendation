from src.diversity import DiversityReranker


def movie(movie_id, score, genres=None, keywords=None, collection_id=None):
    return {
        "id": movie_id,
        "score": score,
        "genre_ids": genres or [],
        "keyword_ids": keywords or [],
        "collection_id": collection_id,
    }


def test_reranker_promotes_a_less_redundant_candidate():
    candidates = [
        movie(1, 1.0, [1], [10]),
        movie(2, 0.99, [1], [10]),
        movie(3, 0.95, [2], [20]),
    ]

    result = DiversityReranker(relevance_weight=0.5).rerank(candidates, limit=3)

    assert [item["id"] for item in result] == [1, 3, 2]
    assert result[1]["diversity_penalty"] == 0.0
    assert result[2]["diversity_penalty"] == 1.0


def test_reranker_applies_collection_cap_when_alternatives_exist():
    candidates = [
        movie(1, 1.0, collection_id=10),
        movie(2, 0.99, collection_id=10),
        movie(3, 0.98, collection_id=20),
    ]

    result = DiversityReranker(
        relevance_weight=1.0,
        max_per_collection=1,
    ).rerank(candidates, limit=2)

    assert [item["id"] for item in result] == [1, 3]


def test_reranker_has_stable_ties_with_missing_metadata():
    result = DiversityReranker().rerank(
        [movie(1, 0.5), movie(2, 0.5)],
        limit=2,
    )

    assert [item["id"] for item in result] == [1, 2]


def test_reranker_softens_collection_cap_to_fill_limit():
    candidates = [
        movie(1, 1.0, collection_id=10),
        movie(2, 0.9, collection_id=10),
    ]

    result = DiversityReranker(max_per_collection=1).rerank(candidates, limit=2)

    assert [item["id"] for item in result] == [1, 2]
