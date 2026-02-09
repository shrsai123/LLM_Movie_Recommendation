from langchain_community.vectorstores import FAISS
from src.embeddings import PROJECT_ROOT

def load_retriever(embedding_model, config):
    index_path = str(PROJECT_ROOT / config["retrieval"]["index_path"])
    top_k = config["retrieval"]["top_k"]
    db = FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(
        search_kwargs={"k": top_k}
    )