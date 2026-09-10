import os
import re
from typing import Any

import httpx

TMDB_BASE_URL = "https://api.themoviedb.org/3"


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
        data = await self._get(
            f"/movie/{source_movie['id']}/recommendations",
            {"language": "en-US"},
        )

        return {
            "source_movie": source_movie,
            "recommendations": [
                self._compact_movie(movie) for movie in data.get("results", [])[:limit]
            ],
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

    @staticmethod
    def _compact_movie(movie: dict) -> dict:
        release_date = movie.get("release_date") or ""

        poster_path = movie.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

        return {
            "id": movie.get("id"),
            "title": movie.get("title"),
            "release_year": release_date[:4] or None,
            "overview": movie.get("overview"),
            "vote_average": movie.get("vote_average"),
            "poster_url": poster_url,
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
