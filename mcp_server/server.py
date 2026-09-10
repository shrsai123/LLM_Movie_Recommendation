import json
import logging

from dotenv import load_dotenv
from mcp.server import MCPServer

from mcp_server.tmdb_client import MovieNotFoundError, TMDBClient

load_dotenv()

logger = logging.getLogger(__name__)
mcp = MCPServer("tmdb-movie-server")

tmdb = TMDBClient()


def serialize(data: object) -> str:
    return json.dumps(data, ensure_ascii=False)


@mcp.tool()
async def search_movies(query: str, year: int | None = None, limit: int = 5) -> str:
    """Search TMDB for movies matching a title or phrase."""
    results = await tmdb.search_movies(query, year, limit)
    return serialize(results)


@mcp.tool()
async def get_movie_details(title: str, year: int | None = None) -> str:
    """Get plot, director, cast, genres and runtime for a movie."""
    try:
        details = await tmdb.get_movie_details(title, year)
    except MovieNotFoundError as exc:
        return serialize({"error": str(exc)})
    return serialize(details)


@mcp.tool()
async def get_trending_movies(time_window: str = "week", limit: int = 5) -> str:
    """Get movies currently trending today or this week."""
    results = await tmdb.get_trending_movies(time_window, limit)
    return serialize(results)


@mcp.tool()
async def get_similar_movies(
    title: str,
    year: int | None = None,
    limit: int = 5,
) -> str:
    """Get TMDB recommendations related to a specified movie."""
    try:
        result = await tmdb.get_similar_movies(title, year, limit)
    except MovieNotFoundError as exc:
        return serialize({"error": str(exc)})
    return serialize(result)


@mcp.tool()
async def get_watch_providers(
    title: str,
    region: str = "US",
    year: int | None = None,
) -> str:
    """Find streaming, rental and purchase providers for a movie."""
    try:
        result = await tmdb.get_watch_providers(title, region, year)
    except MovieNotFoundError as exc:
        return serialize({"error": str(exc)})
    return serialize(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
