import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

TMDB_BASE_URL = "https://api.themoviedb.org/3"

load_dotenv()


class TMDBError(Exception):
    pass


class MovieNotFoundError(TMDBError):
    pass


class TMDBClient:
    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("TMDB_BEARER_TOKEN")
        if not self.token:
            raise ValueError("TMDB Bearer token is required. Please set it in the .env file.")

    async def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(
            base_url=TMDB_BASE_URL,
            headers=headers,
            timeout=10.0,
        ) as client:
            response = await client.get(path, params=params)
            response.raise_for_status()
            return response.json()

    async def search_movies(
        self, query: str, year: int | None = None, limit: int = 5
    ) -> list[dict]:
        params = {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
        }
        if year:
            params["year"] = year

        data = await self._get("/search/movie", params)
        return [self._compact_movie(movie) for movie in data.get("results", [])[:limit]]

    async def resolve_movie(self, title: str, year: int | None = None) -> dict:
        results = await self.search_movies(title, year)
        if not results:
            raise MovieNotFoundError(f"No movie found for: {title}")

        normalized_title = self._normalize_title(title)
        for movie in results:
            if self._normalize_title(movie.get("title", "")) == normalized_title:
                return movie

        return results[0]

    async def get_movie_details(self, title: str, year: int | None = None) -> dict:
        movie = await self.resolve_movie(title, year)
        data = await self._get(f"/movie/{movie['id']}", {"append_to_response": "credits"})
        return self._detailed_movie(data)

    async def get_trending_movies(
        self,
        time_window: str = "week",
        limit: int = 5,
    ) -> list[dict]:
        if time_window not in {"day", "week"}:
            raise ValueError("time_window must be 'day' or 'week'")

        data = await self._get(
            f"/trending/movie/{time_window}",
            {"language": "en-US"},
        )

        return [self._compact_movie(movie) for movie in data.get("results", [])[:limit]]

    async def get_similar_movies(
        self,
        title: str,
        year: int | None = None,
        limit: int = 5,
    ) -> dict:
        source_movie = await self.resolve_movie(title, year)
        source_movie = (await self.enrich_movies([source_movie]))[0]
        data = await self._get(
            f"/movie/{source_movie['id']}/recommendations",
            {"language": "en-US"},
        )
        recommendations = [
            self._compact_movie(movie) for movie in data.get("results", [])[:limit]
        ]
        recommendations = await self.enrich_movies(recommendations)

        return {
            "source_movie": source_movie,
            "recommendations": recommendations,
        }

    async def get_watch_providers(
        self,
        title: str,
        region: str = "US",
        year: int | None = None,
    ) -> dict:
        movie = await self.resolve_movie(title, year)
        data = await self._get(f"/movie/{movie['id']}/watch/providers")

        region_data = data.get("results", {}).get(region.upper(), {})

        return {
            "movie": movie,
            "region": region.upper(),
            "stream": self._provider_names(region_data.get("flatrate", [])),
            "rent": self._provider_names(region_data.get("rent", [])),
            "buy": self._provider_names(region_data.get("buy", [])),
            "tmdb_link": region_data.get("link"),
        }

    async def enrich_movies(self, movies: list[dict]) -> list[dict]:
        """Fetch ranking metadata for known TMDB movie IDs."""
        enriched = []
        for movie in movies:
            movie_id = movie.get("id")
            if movie_id is None:
                enriched.append(self._with_empty_ranking_metadata(movie))
                continue

            data = await self._get(
                f"/movie/{movie_id}",
                {"append_to_response": "keywords", "language": "en-US"},
            )
            enriched.append(
                {
                    **movie,
                    **self._compact_movie(data),
                    **self._ranking_metadata(data),
                }
            )
        return enriched

    @staticmethod
    def _compact_movie(movie: dict) -> dict:
        release_date = movie.get("release_date") or ""

        poster_path = movie.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        genre_ids = movie.get("genre_ids")
        if genre_ids is None:
            genre_ids = [
                genre["id"] for genre in movie.get("genres", []) if genre.get("id") is not None
            ]

        return {
            "id": movie.get("id"),
            "title": movie.get("title"),
            "release_year": release_date[:4] or None,
            "overview": movie.get("overview"),
            "genre_ids": genre_ids,
            "vote_average": movie.get("vote_average"),
            "poster_url": poster_url,
        }

    @staticmethod
    def _ranking_metadata(movie: dict) -> dict:
        keywords = movie.get("keywords", {}).get("keywords", [])
        collection = movie.get("belongs_to_collection") or {}
        return {
            "keyword_ids": [keyword["id"] for keyword in keywords if keyword.get("id")],
            "keywords": [keyword["name"] for keyword in keywords if keyword.get("name")],
            "collection_id": collection.get("id"),
            "collection_name": collection.get("name"),
            "ranking_metadata_available": True,
        }

    @staticmethod
    def _with_empty_ranking_metadata(movie: dict) -> dict:
        return {
            **movie,
            "keyword_ids": movie.get("keyword_ids", []),
            "keywords": movie.get("keywords", []),
            "collection_id": movie.get("collection_id"),
            "collection_name": movie.get("collection_name"),
            "ranking_metadata_available": movie.get("ranking_metadata_available", False),
        }

    @classmethod
    def _detailed_movie(cls, movie: dict) -> dict:
        result = cls._compact_movie(movie)

        crew = movie.get("credits", {}).get("crew", [])
        cast = movie.get("credits", {}).get("cast", [])

        director = next(
            (person.get("name") for person in crew if person.get("job") == "Director"),
            None,
        )

        result.update(
            {
                "director": director,
                "cast": [person.get("name") for person in cast[:5]],
                "genres": [genre.get("name") for genre in movie.get("genres", [])],
                "runtime": movie.get("runtime"),
            }
        )

        return result

    @staticmethod
    def _provider_names(providers: list[dict]) -> list[str]:
        return [
            provider["provider_name"] for provider in providers if provider.get("provider_name")
        ]

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", title.lower())
