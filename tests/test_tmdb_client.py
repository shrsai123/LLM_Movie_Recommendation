import asyncio
from unittest.mock import AsyncMock

from mcp_server.tmdb_client import TMDBClient


def test_similar_movies_enriches_source_and_candidates_with_ranking_metadata():
    client = TMDBClient.__new__(TMDBClient)
    client.resolve_movie = AsyncMock(
        return_value={
            "id": 1,
            "title": "Source",
            "overview": "A source movie",
            "genre_ids": [878],
        }
    )
    client._get = AsyncMock(
        side_effect=[
            {
                "id": 1,
                "title": "Source",
                "overview": "A source movie",
                "genres": [{"id": 878, "name": "Science Fiction"}],
                "keywords": {
                    "keywords": [
                        {"id": 100, "name": "space travel"},
                        {"id": 101, "name": "astronaut"},
                    ]
                },
                "belongs_to_collection": {"id": 10, "name": "Source Collection"},
            },
            {
                "results": [
                    {
                        "id": 2,
                        "title": "Candidate",
                        "overview": "A candidate movie",
                        "genre_ids": [878],
                    }
                ]
            },
            {
                "id": 2,
                "title": "Candidate",
                "overview": "A candidate movie",
                "genres": [{"id": 878, "name": "Science Fiction"}],
                "keywords": {
                    "keywords": [{"id": 200, "name": "deep space"}]
                },
                "belongs_to_collection": None,
            },
        ]
    )

    result = asyncio.run(client.get_similar_movies("Source"))

    assert result["source_movie"]["keywords"] == ["space travel", "astronaut"]
    assert result["source_movie"]["keyword_ids"] == [100, 101]
    assert result["source_movie"]["collection_id"] == 10
    assert result["source_movie"]["genre_ids"] == [878]
    assert result["recommendations"][0]["keywords"] == ["deep space"]
    assert result["recommendations"][0]["keyword_ids"] == [200]
    assert result["recommendations"][0]["collection_id"] is None
    detail_call = client._get.await_args_list[0]
    assert detail_call.args[0] == "/movie/1"
    assert detail_call.args[1]["append_to_response"] == "keywords"
    candidate_call = client._get.await_args_list[2]
    assert candidate_call.args[0] == "/movie/2"
    assert candidate_call.args[1]["append_to_response"] == "keywords"
