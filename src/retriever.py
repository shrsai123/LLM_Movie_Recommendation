from langchain_community.vectorstores import FAISS

from src.embeddings import PROJECT_ROOT


def load_vector_store(embedding_model, config):
    index_path = str(PROJECT_ROOT / config["retrieval"]["index_path"])
    return FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
