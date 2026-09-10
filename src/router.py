# src/router.py

import re
from enum import Enum


class Intent(str, Enum):
    TRENDING = "trending"
    WATCH_PROVIDERS = "watch_providers"
    MOVIE_DETAILS = "movie_details"
    SIMILAR_MOVIES = "similar_movies"
    RAG = "rag"


def classify_intent(query: str) -> Intent:
    normalized = query.lower().strip()

    if any(term in normalized for term in ["trending", "popular today", "popular this week"]):
        return Intent.TRENDING

    if any(term in normalized for term in ["where can i watch", "streaming on", "watch provider"]):
        return Intent.WATCH_PROVIDERS

    if any(term in normalized for term in ["similar to", "movies like"]):
        return Intent.SIMILAR_MOVIES

    if any(term in normalized for term in ["details about", "cast of", "who directed"]):
        return Intent.MOVIE_DETAILS

    return Intent.RAG


def extract_title(query: str, intent: Intent) -> str | None:
    patterns = {
        Intent.WATCH_PROVIDERS: (r"(?:where can i watch|watch|streaming on)\s+(.+?)[?.]*$"),
        Intent.SIMILAR_MOVIES: (r"(?:similar to|movies like)\s+(.+?)[?.]*$"),
        Intent.MOVIE_DETAILS: (r"(?:details about|cast of|who directed)\s+(.+?)[?.]*$"),
    }

    pattern = patterns.get(intent)

    if pattern is None:
        return None

    match = re.search(pattern, query, flags=re.IGNORECASE)

    if match is None:
        return None

    title = match.group(1).strip(" \"'?.")
    return re.sub(r"\s*\(?\b(?:19|20)\d{2}\b\)?\s*$", "", title).strip(" \"'?.")


def extract_year(query: str) -> int | None:
    match = re.search(r"\b((?:19|20)\d{2})\b", query)
    return int(match.group(1)) if match else None
