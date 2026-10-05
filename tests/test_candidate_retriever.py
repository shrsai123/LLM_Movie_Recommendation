from src.candidate_retriever import FaissMovieCandidateRetriever


class FakeDocument:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def similarity_search(self, query, k):
        self.calls.append({"query": query, "k": k})
        return self.documents[:k]


def test_search_query_retrieves_faiss_candidates_from_text():
    vector_store = FakeVectorStore(
        [
            FakeDocument({"id": 1, "title": "Arrival"}),
            FakeDocument({"id": 2, "title": ""}),
            FakeDocument({"id": 3, "title": "Contact"}),
        ]
    )
    retriever = FaissMovieCandidateRetriever(vector_store)

    results = retriever.search_query("thoughtful alien contact", limit=3)

    assert vector_store.calls == [{"query": "thoughtful alien contact", "k": 3}]
    assert results == [
        {"id": 1, "title": "Arrival", "candidate_sources": ["faiss"]},
        {"id": 3, "title": "Contact", "candidate_sources": ["faiss"]},
    ]


def test_search_excludes_source_movie():
    vector_store = FakeVectorStore(
        [
            FakeDocument({"id": 10, "title": "Interstellar"}),
            FakeDocument({"id": 11, "title": "Ad Astra"}),
        ]
    )
    retriever = FaissMovieCandidateRetriever(vector_store)

    results = retriever.search({"id": 10, "title": "Interstellar"}, limit=1)

    assert vector_store.calls == [{"query": "Interstellar", "k": 2}]
    assert results == [{"id": 11, "title": "Ad Astra", "candidate_sources": ["faiss"]}]
