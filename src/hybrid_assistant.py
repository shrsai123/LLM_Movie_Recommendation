import asyncio

from src.mcp_client import MovieMCPClient, MovieMCPError
from src.router import Intent, classify_intent, extract_title, extract_year

class HybridMovieAssistant:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.mcp = MovieMCPClient()

    async def answer(self, query: str) -> dict:
        intent = classify_intent(query)

        if intent == Intent.RAG:
            result = await asyncio.to_thread(
                self.qa_chain.invoke,
                {"query": query},
            )

            return {
                "answer": result["result"],
                "route": "rag",
                "tools_used": [],
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
                    "tools_used": [tool],
                }

        else:
            title = extract_title(query, intent)

            if not title:
                return {
                    "answer": "Which movie are you referring to?",
                    "route": "clarification",
                    "tools_used": [],
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
                arguments["region"] = "US"

            try:
                data = await self.mcp.call_tool(tool, arguments)
            except MovieMCPError as exc:
                return {
                    "answer": f"I couldn't get that from TMDB right now. {exc}",
                    "route": "mcp_error",
                    "tools_used": [tool],
                }

        return {
            "answer": self._format_result(tool, data),
            "route": "mcp",
            "tools_used": [tool],
        }

    @staticmethod
    def _format_result(tool: str, data: dict | list) -> str:
        # Move this logic into formatters.py after the first version works.
        if isinstance(data, dict) and data.get("error"):
            return data["error"]

        if tool == "get_trending_movies":
            lines = [
                f"- {movie['title']} ({movie.get('release_year', 'Unknown')})"
                for movie in data
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
                for movie in recommendations
            ]
            return f"Movies similar to {source}:\n" + "\n".join(lines)

        return "I got a response, but I do not know how to format it yet."

    @staticmethod
    def _movie_label(movie: dict) -> str:
        title = movie.get("title", "Unknown title")
        year = movie.get("release_year")
        return f"{title} ({year})" if year else title
