from src.recommender import MovieRecommender


class FakeEmbeddings:
    vectors = {
        "space survival": [1.0, 0.0],
        "astronaut, survival": [1.0, 0.0],
        "cosmonaut, stranded": [0.8, 0.6],
        "courtroom, lawyer": [0.0, 1.0],
    }

    def embed_query(self, text):
        return self.vectors[text.lower()]

    def embed_documents(self, texts):
        return [self.vectors[text.lower()] for text in texts]


def test_recommender_accepts_ranking_weights_and_exposes_signals():
    recommender = MovieRecommender(
        FakeEmbeddings(),
        weights={
            "overview": 0.6,
            "keywords": 0.25,
            "genres": 0.1,
            "collection": 0.05,
        },
    )

    ranked = recommender.rank(
        {
            "id": 1,
            "overview": "Space survival",
            "genre_ids": [878, 18],
            "keyword_ids": [1, 2],
            "keywords": ["astronaut", "survival"],
            "collection_id": 10,
        },
        [
            {
                "id": 2,
                "title": "Candidate",
                "overview": "Space survival",
                "genre_ids": [878],
                "keyword_ids": [2, 3],
                "keywords": ["cosmonaut", "stranded"],
                "collection_id": 10,
            }
        ],
    )

    assert ranked[0]["signals"] == {
        "overview": 1.0,
        "keywords": 0.8,
        "genres": 1 / 2,
        "collection": 1.0,
    }
    assert ranked[0]["score"] == 0.9


def test_keyword_signal_uses_meaning_instead_of_id_overlap():
    recommender = MovieRecommender(
        FakeEmbeddings(),
        weights={"overview": 0, "keywords": 1, "genres": 0, "collection": 0},
    )

    ranked = recommender.rank(
        {
            "id": 1,
            "keyword_ids": [1, 2],
            "keywords": ["astronaut", "survival"],
        },
        [
            {
                "id": 2,
                "title": "Semantically related",
                "keyword_ids": [3, 4],
                "keywords": ["cosmonaut", "stranded"],
            },
            {
                "id": 3,
                "title": "Unrelated",
                "keyword_ids": [1, 2],
                "keywords": ["courtroom", "lawyer"],
            },
        ],
    )

    assert [movie["title"] for movie in ranked] == ["Semantically related", "Unrelated"]
    assert ranked[0]["signals"]["keywords"] == 0.8
    assert ranked[1]["signals"]["keywords"] == 0.0


def test_keyword_signal_is_unavailable_without_keyword_names():
    recommender = MovieRecommender(
        FakeEmbeddings(),
        weights={"overview": 0, "keywords": 1, "genres": 0, "collection": 0},
    )

    ranked = recommender.rank(
        {"id": 1, "keyword_ids": [1]},
        [{"id": 2, "keyword_ids": [1]}],
    )

    assert ranked[0]["signals"]["keywords"] is None
    assert ranked[0]["score"] == 0.0
