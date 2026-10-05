# src/router.py

import re
from enum import Enum


class Intent(str, Enum):
    TRENDING = "trending"
    WATCH_PROVIDERS = "watch_providers"
    MOVIE_DETAILS = "movie_details"
    MOVIE_ENDING = "movie_ending"
    SIMILAR_MOVIES = "similar_movies"
    RECOMMENDATION = "recommendation"
    GENERAL = "general"


def classify_intent(query: str) -> Intent:
    normalized = query.lower().strip().lstrip("/").strip()

    if any(term in normalized for term in ["trending", "popular today", "popular this week"]):
        return Intent.TRENDING

    if any(term in normalized for term in ["where can i watch", "streaming on", "watch provider"]):
        return Intent.WATCH_PROVIDERS

    if any(
        term in normalized
        for term in ["similar to", "movies like", "recommendations for", "other movies"]
    ):
        return Intent.SIMILAR_MOVIES

    if any(term in normalized for term in ["ending of", "ending for"]):
        return Intent.MOVIE_ENDING

    if any(
        term in normalized
        for term in [
            "details about",
            "cast of",
            "who directed",
            "plot of",
            "plot for",
            "synopsis of",
            "synopsis for",
            "what happens in",
        ]
    ):
        return Intent.MOVIE_DETAILS

    if any(
        term in normalized for term in ["recommend", "suggest", "looking for", "in the mood for"]
    ) or any(
        re.search(pattern, normalized)
        for pattern in [
            r"\bgive me (?:all |some )?movies\b",
            r"\bmovies that (?:have|has)\b",
            r"\bi want (?:all |some )?movies\b",
        ]
    ):
        return Intent.RECOMMENDATION

    return Intent.GENERAL


def extract_title(query: str, intent: Intent) -> str | None:
    query = query.strip().lstrip("/").strip()
    patterns = {
        Intent.WATCH_PROVIDERS: (r"(?:where can i watch|watch|streaming on)\s+(.+?)[?.]*$"),
        Intent.SIMILAR_MOVIES: (r"(?:similar to|movies like)\s+(.+?)[?.]*$"),
        Intent.MOVIE_DETAILS: (
            r"(?:details about|cast of|who directed|"
            r"(?:(?:what(?:'s|s| is)\s+)?(?:the\s+)?)?"
            r"(?:plot|story|synopsis)\s+(?:of|for)|"
            r"what happens in)\s+(.+?)[?.]*$"
        ),
        Intent.MOVIE_ENDING: (
            r"(?:(?:what(?:'s|s| is)\s+)?(?:the\s+)?)?"
            r"ending\s+(?:of|for)\s+(.+?)[?.]*$"
        ),
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


def extract_search_query(query: str) -> str:
    """Reduce a general request to the phrase sent to TMDB movie search."""
    query = query.strip().lstrip("/").strip()
    search_query = re.sub(
        r"^(?:please\s+)?(?:search(?:\s+for)?|find|look up|tell me about|"
        r"show me|explain|what is|what's|whats)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )
    search_query = re.sub(
        r"\s*\(?\b(?:19|20)\d{2}\b\)?\s*[?.]*$",
        "",
        search_query,
    )
    return search_query.strip(" \"'?.") or query
