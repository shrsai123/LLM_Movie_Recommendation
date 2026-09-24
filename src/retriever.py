from langchain_community.vectorstores import FAISS

from src.embeddings import PROJECT_ROOT


def load_vector_store(embedding_model, config):
    index_path = str(PROJECT_ROOT / config["retrieval"]["index_path"])
    return FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def build_retriever(vector_store, config):
    top_k = config["retrieval"]["top_k"]
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def load_retriever(embedding_model, config):
    """Compatibility helper for callers that need only the RAG retriever."""
    return build_retriever(load_vector_store(embedding_model, config), config)