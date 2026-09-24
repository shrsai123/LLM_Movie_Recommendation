"""Diversify an already relevance-ranked list of movie candidates."""

from collections import Counter


def _jaccard(left: list | None, right: list | None) -> float | None:
    left_values = set(left or [])
    right_values = set(right or [])
    if not left_values or not right_values:
        return None
    return len(left_values & right_values) / len(left_values | right_values)


def _candidate_similarity(left: dict, right: dict) -> float:
    """Estimate redundancy using metadata, not source-movie relevance."""
    weighted_signals = []

    genre_similarity = _jaccard(left.get("genre_ids"), right.get("genre_ids"))
    if genre_similarity is not None:
        weighted_signals.append((0.50, genre_similarity))

    keyword_similarity = _jaccard(left.get("keyword_ids"), right.get("keyword_ids"))
    if keyword_similarity is None:
        keyword_similarity = _jaccard(left.get("keywords"), right.get("keywords"))
    if keyword_similarity is not None:
        weighted_signals.append((0.30, keyword_similarity))

    left_collection = left.get("collection_id")
    right_collection = right.get("collection_id")
    if left_collection is not None and right_collection is not None:
        weighted_signals.append((0.20, float(left_collection == right_collection)))

    total_weight = sum(weight for weight, _ in weighted_signals)
    if not total_weight:
        return 0.0
    return sum(weight * value for weight, value in weighted_signals) / total_weight


class DiversityReranker:
    """Apply maximal-marginal-relevance selection to ranked movies."""

    def __init__(
        self,
        relevance_weight: float = 0.85,
        max_per_collection: int = 2,
    ):
        if not 0 <= relevance_weight <= 1:
            raise ValueError("relevance_weight must be between 0 and 1")
        if max_per_collection < 1:
            raise ValueError("max_per_collection must be positive")
        self.relevance_weight = relevance_weight
        self.max_per_collection = max_per_collection

    def rerank(self, candidates: list[dict], limit: int = 5) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not candidates:
            return []

        remaining = [
            {**candidate, "relevance_rank": rank}
            for rank, candidate in enumerate(candidates, start=1)
        ]
        selected = []
        collection_counts = Counter()

        while remaining and len(selected) < limit:
            eligible = [
                movie
                for movie in remaining
                if movie.get("collection_id") is None
                or collection_counts[movie["collection_id"]] < self.max_per_collection
            ]
            if not eligible:
                eligible = remaining

            best_movie = None
            best_key = None
            best_penalty = 0.0
            best_selection_score = 0.0

            for movie in eligible:
                redundancy = (
                    max(_candidate_similarity(movie, chosen) for chosen in selected)
                    if selected
                    else 0.0
                )
                relevance = float(movie.get("score") or 0.0)
                selection_score = (
                    self.relevance_weight * relevance
                    - (1 - self.relevance_weight) * redundancy
                )
                key = (selection_score, relevance, -movie["relevance_rank"])
                if best_key is None or key > best_key:
                    best_movie = movie
                    best_key = key
                    best_penalty = redundancy
                    best_selection_score = selection_score

            remaining.remove(best_movie)
            best_movie = {
                **best_movie,
                "diversity_penalty": round(best_penalty, 3),
                "diversity_score": round(best_selection_score, 3),
            }
            selected.append(best_movie)
            collection_id = best_movie.get("collection_id")
            if collection_id is not None:
                collection_counts[collection_id] += 1

        return selected
