# scripts/build_index.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from langchain_community.vectorstores import FAISS

from src.embeddings import PROJECT_ROOT, get_embedding_model, load_config
from src.preprocessing import run_preprocessing

config = load_config()
docs = run_preprocessing(config)
embeddings = get_embedding_model(config)
index_path = str(PROJECT_ROOT / config["retrieval"]["index_path"])

db = FAISS.from_documents(docs, embeddings)
db.save_local(index_path)
print(f"Index built with {len(docs)} documents")
