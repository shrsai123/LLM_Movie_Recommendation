"""Rank movies against a natural-language preference query."""

import math

from src.recommender import _cosine_similarity

DEFAULT_PREFERENCE_WEIGHTS = {
    "query": 0.60,
    "keywords": 0.10,
    "genres": 0.20,
    "quality": 0.10,
}


class PreferenceRecommender:
    def __init__(self, embeddings, weights: dict[str, float] | None = None):
        self.embeddings = embeddings
        self.weights = dict(DEFAULT_PREFERENCE_WEIGHTS if weights is None else weights)
        if set(self.weights) != set(DEFAULT_PREFERENCE_WEIGHTS):
            raise ValueError(
                f"weights must contain exactly: {', '.join(DEFAULT_PREFERENCE_WEIGHTS)}"
            )
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("preference ranking weights cannot be negative")
        if not math.isclose(sum(self.weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("preference ranking weights must sum to 1.0")

    def rank(
        self,
        preferences: dict,
        candidates: list[dict],
        limit: int = 5,
    ) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")

        unique = []
        seen_ids = set()
        for candidate in candidates:
            movie_id = candidate.get("id")
            if movie_id is not None and movie_id in seen_ids:
                continue
            if movie_id is not None:
                seen_ids.add(movie_id)
            unique.append(candidate)

        query = preferences.get("query", "").strip()
        query_vector = self.embeddings.embed_query(query) if query and unique else None
        movie_texts = [self._movie_text(movie) for movie in unique]
        movie_vectors = self.embeddings.embed_documents(movie_texts) if unique else []

        keyword_indexes = [
            (index, ", ".join(movie.get("keywords") or []))
            for index, movie in enumerate(unique)
            if movie.get("keywords")
        ]
        keyword_scores = {}
        if query_vector is not None and keyword_indexes:
            vectors = self.embeddings.embed_documents([text for _, text in keyword_indexes])
            keyword_scores = {
                index: _cosine_similarity(query_vector, vector)
                for (index, _), vector in zip(keyword_indexes, vectors, strict=True)
            }

        requested_genres = set(preferences.get("genre_ids") or [])
        ranked = []
        for index, (movie, vector) in enumerate(zip(unique, movie_vectors, strict=True)):
            query_score = (
                _cosine_similarity(query_vector, vector) if query_vector is not None else None
            )
            keyword_score = keyword_scores.get(index)
            movie_genres = set(movie.get("genre_ids") or [])
            genre_score = (
                len(requested_genres & movie_genres) / len(requested_genres)
                if requested_genres
                else None
            )
            vote_average = movie.get("vote_average")
            quality_score = (
                max(0.0, min(1.0, float(vote_average) / 10.0))
                if vote_average is not None
                else None
            )
            signals = {
                "query": query_score,
                "keywords": keyword_score,
                "genres": genre_score,
                "quality": quality_score,
            }
            available = [
                (self.weights[name], value)
                for name, value in signals.items()
                if value is not None and self.weights[name] > 0
            ]
            total_weight = sum(weight for weight, _ in available)
            score = (
                sum(weight * value for weight, value in available) / total_weight
                if total_weight
                else 0.0
            )
            matched = []
            if query_score is not None and query_score > 0:
                matched.append("matches your description")
            if genre_score is not None and genre_score > 0:
                matched.append("requested genre")
            if keyword_score is not None and keyword_score > 0:
                matched.append("related themes")
            reason = ", ".join(matched) if matched else "discovered from your preferences"
            ranked.append(
                (
                    score,
                    index,
                    {
                        **movie,
                        "score": round(score, 3),
                        "reason": reason.capitalize(),
                        "signals": {
                            name: round(value, 3) if value is not None else None
                            for name, value in signals.items()
                        },
                    },
                )
            )

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [movie for _, _, movie in ranked[:limit]]

    @staticmethod
    def _movie_text(movie: dict) -> str:
        sections = [
            movie.get("title") or "",
            movie.get("overview") or "",
            ", ".join(movie.get("keywords") or []),
            ", ".join(movie.get("genres") or []),
        ]
        return "\n".join(section for section in sections if section).strip()
