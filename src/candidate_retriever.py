"""Retrieve structured movie candidates from the local FAISS index."""


class FaissMovieCandidateRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def search(self, source_movie: dict, limit: int = 20) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")

        query = self._query_text(source_movie)
        # Request one extra result because the source movie may retrieve itself.
        documents = self.vector_store.similarity_search(query, k=limit + 1)
        source_id = source_movie.get("id")
        return self._candidates_from_documents(
            documents,
            limit=limit,
            excluded_movie_id=source_id,
        )

    def search_query(self, query: str, limit: int = 20) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")

        query = query.strip()
        if not query:
            return []

        documents = self.vector_store.similarity_search(query, k=limit)
        return self._candidates_from_documents(documents, limit=limit)

    @staticmethod
    def _candidates_from_documents(
        documents,
        limit: int,
        excluded_movie_id=None,
    ) -> list[dict]:
        candidates = []

        for document in documents:
            movie = dict(document.metadata)
            if excluded_movie_id is not None and movie.get("id") == excluded_movie_id:
                continue
            if not movie.get("title"):
                continue
            movie["candidate_sources"] = ["faiss"]
            candidates.append(movie)
            if len(candidates) == limit:
                break

        return candidates

    @staticmethod
    def _query_text(movie: dict) -> str:
        sections = [
            movie.get("title") or "",
            movie.get("overview") or "",
            ", ".join(movie.get("keywords") or []),
            ", ".join(str(value) for value in movie.get("genres") or []),
        ]
        return "\n".join(section for section in sections if section).strip()
