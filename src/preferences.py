"""Deterministic preference extraction for discovery recommendations."""

import re

GENRES = {
    "action": 28,
    "adventure": 12,
    "animation": 16,
    "animated": 16,
    "comedy": 35,
    "funny": 35,
    "crime": 80,
    "documentary": 99,
    "drama": 18,
    "family": 10751,
    "fantasy": 14,
    "history": 36,
    "historical": 36,
    "horror": 27,
    "music": 10402,
    "musical": 10402,
    "mystery": 9648,
    "romance": 10749,
    "romantic": 10749,
    "science fiction": 878,
    "science-fiction": 878,
    "sci-fi": 878,
    "sci fi": 878,
    "thriller": 53,
    "war": 10752,
    "western": 37,
}


def extract_preferences(query: str) -> dict:
    normalized = query.lower()
    found = []
    seen_ids = set()

    # Match longer aliases first so "science fiction" is not partially lost.
    for name, genre_id in sorted(GENRES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", normalized):
            if genre_id not in seen_ids:
                found.append({"id": genre_id, "name": _canonical_genre_name(genre_id)})
                seen_ids.add(genre_id)

    year_min, year_max = _extract_year_bounds(normalized)
    return {
        "query": query.strip(),
        "genre_ids": [genre["id"] for genre in found],
        "genres": [genre["name"] for genre in found],
        "year_min": year_min,
        "year_max": year_max,
    }


def _extract_year_bounds(query: str) -> tuple[int | None, int | None]:
    decade = re.search(r"\b((?:19|20)\d0)s\b", query)
    if decade:
        start = int(decade.group(1))
        return start, start + 9

    after = re.search(r"\b(?:after|since)\s+((?:19|20)\d{2})\b", query)
    before = re.search(r"\bbefore\s+((?:19|20)\d{2})\b", query)
    if after or before:
        return (
            int(after.group(1)) if after else None,
            int(before.group(1)) if before else None,
        )

    year = re.search(r"\b((?:19|20)\d{2})\b", query)
    if year:
        value = int(year.group(1))
        return value, value
    return None, None


def _canonical_genre_name(genre_id: int) -> str:
    names = {
        28: "Action",
        12: "Adventure",
        16: "Animation",
        35: "Comedy",
        80: "Crime",
        99: "Documentary",
        18: "Drama",
        10751: "Family",
        14: "Fantasy",
        36: "History",
        27: "Horror",
        10402: "Music",
        9648: "Mystery",
        10749: "Romance",
        878: "Science Fiction",
        53: "Thriller",
        10752: "War",
        37: "Western",
    }
    return names[genre_id]
