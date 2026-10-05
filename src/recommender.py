"""Rank TMDB candidates using the shared movie embedding model."""

import math

DEFAULT_SIGNAL_WEIGHTS = {
    "overview": 0.60,
    "keywords": 0.25,
    "genres": 0.10,
    "collection": 0.05,
}
KNOWN_SIGNALS = set(DEFAULT_SIGNAL_WEIGHTS)

GENRE_NAMES = {
    12: "Adventure",
    14: "Fantasy",
    16: "Animation",
    18: "Drama",
    27: "Horror",
    28: "Action",
    35: "Comedy",
    36: "History",
    37: "Western",
    53: "Thriller",
    80: "Crime",
    99: "Documentary",
    878: "Science Fiction",
    9648: "Mystery",
    10402: "Music",
    10749: "Romance",
    10751: "Family",
    10752: "War",
}


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Movie embeddings must have matching dimensions")

    magnitude = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if not magnitude:
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right)) / magnitude))


def _genre_similarity(source: dict, candidate: dict) -> float | None:
    source_genres = set(source.get("genre_ids") or [])
    candidate_genres = set(candidate.get("genre_ids") or [])
    if not source_genres or not candidate_genres:
        return None
    return len(source_genres & candidate_genres) / len(source_genres | candidate_genres)


def _keyword_text(movie: dict) -> str:
    keywords = {
        str(keyword).strip().lower()
        for keyword in movie.get("keywords") or []
        if str(keyword).strip()
    }
    return ", ".join(sorted(keywords))


def _collection_similarity(source: dict, candidate: dict) -> float | None:
    source_collection = source.get("collection_id")
    if not source_collection:
        return None
    return 1.0 if source_collection == candidate.get("collection_id") else 0.0


def _genre_names(movie: dict) -> list[str]:
    names = []
    for genre in movie.get("genres") or []:
        name = genre.get("name") if isinstance(genre, dict) else str(genre)
        if name and name not in names:
            names.append(name)
    if names:
        return names
    return [
        GENRE_NAMES[genre_id]
        for genre_id in movie.get("genre_ids") or []
        if genre_id in GENRE_NAMES
    ]


def _shared_text(left: list[str], right: list[str]) -> list[str]:
    left_values = {value.casefold() for value in left}
    return [value for value in right if value.casefold() in left_values]


def _format_items(items: list[str], limit: int = 3) -> str:
    selected = items[:limit]
    if len(selected) == 1:
        return selected[0]
    if len(selected) == 2:
        return f"{selected[0]} and {selected[1]}"
    return f"{', '.join(selected[:-1])}, and {selected[-1]}"


def _build_comparison(source: dict, candidate: dict, signals: dict) -> str:
    """Turn ranking metadata into a source-specific explanation."""
    source_title = source.get("title") or "the source movie"
    candidate_title = candidate.get("title") or "This movie"
    shared_genres = _shared_text(_genre_names(source), _genre_names(candidate))
    shared_keywords = _shared_text(
        [str(value) for value in source.get("keywords") or []],
        [str(value) for value in candidate.get("keywords") or []],
    )

    similarities = []
    if shared_genres:
        genre_label = "genre" if len(shared_genres) == 1 else "genres"
        similarities.append(f"the {_format_items(shared_genres)} {genre_label}")
    if shared_keywords:
        similarities.append(f"themes involving {_format_items(shared_keywords)}")
    if (signals.get("collection") or 0) > 0:
        similarities.append("the same collection or franchise")

    if similarities:
        comparison = (
            f"Compared with {source_title}, {candidate_title} shares {_format_items(similarities)}."
        )
    else:
        comparison = (
            f"{candidate_title} was matched with {source_title} because their plots "
            "and themes are semantically similar."
        )

    overview = str(candidate.get("overview") or "").strip()
    if overview and (signals.get("overview") or 0) > 0:
        comparison += f" Its premise—{overview}—is also close in narrative focus."
    elif (signals.get("overview") or 0) > 0:
        comparison += " Their story premises also have a similar narrative focus."
    return comparison


def _validate_weights(weights: dict[str, float] | None) -> dict[str, float]:
    if weights is None:
        return dict(DEFAULT_SIGNAL_WEIGHTS)

    unknown = set(weights) - KNOWN_SIGNALS
    if unknown:
        raise ValueError(f"Unknown ranking signal weights: {', '.join(sorted(unknown))}")

    normalized = {signal: 0.0 for signal in KNOWN_SIGNALS}
    for signal, weight in weights.items():
        if not isinstance(weight, int | float) or weight < 0:
            raise ValueError("Ranking signal weights must be non-negative numbers")
        normalized[signal] = float(weight)

    if not any(normalized.values()):
        raise ValueError("At least one ranking signal weight must be greater than zero")
    return normalized


class MovieRecommender:
    """Rerank candidates; TMDB remains responsible for candidate discovery."""

    def __init__(
        self,
        embeddings,
        semantic_weight: float = 0.8,
        weights: dict[str, float] | None = None,
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        self.embeddings = embeddings
        self.weights = _validate_weights(weights)
        if weights is None:
            self.weights["overview"] = semantic_weight
            self.weights["genres"] = 1 - semantic_weight

    def rank(self, source: dict, candidates: list[dict], limit: int = 5) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")

        # TMDB IDs prevent returning the source or repeated candidates.
        unique = []
        seen_ids = {source.get("id")}
        for candidate in candidates:
            movie_id = candidate.get("id")
            if movie_id is not None and movie_id in seen_ids:
                continue
            if movie_id is not None:
                seen_ids.add(movie_id)
            unique.append(candidate)

        source_overview = (source.get("overview") or "").strip()
        plot_candidates = [
            (index, (movie.get("overview") or "").strip())
            for index, movie in enumerate(unique)
            if (movie.get("overview") or "").strip()
        ]
        plot_scores = {}
        if source_overview and plot_candidates:
            source_vector = self.embeddings.embed_query(source_overview)
            vectors = self.embeddings.embed_documents([plot for _, plot in plot_candidates])
            plot_scores = {
                index: _cosine_similarity(source_vector, vector)
                for (index, _), vector in zip(plot_candidates, vectors, strict=True)
            }

        source_keywords = _keyword_text(source)
        keyword_candidates = []
        for index, movie in enumerate(unique):
            keyword_text = _keyword_text(movie)
            if keyword_text:
                keyword_candidates.append((index, keyword_text))
        keyword_scores = {}
        if source_keywords and keyword_candidates:
            source_vector = self.embeddings.embed_query(source_keywords)
            vectors = self.embeddings.embed_documents(
                [keywords for _, keywords in keyword_candidates]
            )
            keyword_scores = {
                index: _cosine_similarity(source_vector, vector)
                for (index, _), vector in zip(keyword_candidates, vectors, strict=True)
            }

        ranked = []
        for index, movie in enumerate(unique):
            signals = {
                "overview": plot_scores.get(index),
                "keywords": keyword_scores.get(index),
                "genres": _genre_similarity(source, movie),
                "collection": _collection_similarity(source, movie),
            }
            available_scores = [
                (self.weights[signal], value)
                for signal, value in signals.items()
                if value is not None and self.weights[signal] > 0
            ]
            total_weight = sum(weight for weight, _ in available_scores)
            score = (
                sum(weight * value for weight, value in available_scores) / total_weight
                if total_weight
                else 0.0
            )

            reason_labels = {
                "overview": "semantic plot similarity",
                "keywords": "semantic keyword similarity",
                "genres": "genre overlap",
                "collection": "same collection/franchise",
            }
            matched_signals = [
                reason_labels[name]
                for name, value in signals.items()
                if value is not None and value > 0 and self.weights[name] > 0
            ]
            reason = (
                f"Signals: {', '.join(matched_signals)}"
                if matched_signals
                else "Suggested by candidate retrieval"
            )

            ranked.append(
                (
                    score,
                    index,
                    {
                        **movie,
                        "reason": reason,
                        "comparison": _build_comparison(source, movie, signals),
                        "score": round(score, 3),
                        "signals": signals,
                    },
                )
            )

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [movie for _, _, movie in ranked[:limit]]
