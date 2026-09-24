import re


def _candidate_key(movie: dict) -> tuple:
    movie_id = movie.get("id")
    if movie_id is not None:
        return ("id", int(movie_id))

    title = re.sub(r"[^a-z0-9]+", "", (movie.get("title") or "").lower())
    return ("title_year", title, str(movie.get("release_year") or ""))


def merge_candidates(*candidate_groups: tuple[str, list[dict]]) -> list[dict]:
    """Merge candidate groups in order, retaining provenance and filling missing data."""
    merged = []
    positions = {}

    for source_name, candidates in candidate_groups:
        for candidate in candidates:
            key = _candidate_key(candidate)
            if key in positions:
                position = positions[key]
                existing = merged[position]
                for field, value in candidate.items():
                    if existing.get(field) in (None, "", []):
                        existing[field] = value
                sources = existing.setdefault("candidate_sources", [])
                if source_name not in sources:
                    sources.append(source_name)
                continue

            movie = dict(candidate)
            sources = list(movie.get("candidate_sources") or [])
            if source_name not in sources:
                sources.append(source_name)
            movie["candidate_sources"] = sources
            positions[key] = len(merged)
            merged.append(movie)

    return merged
