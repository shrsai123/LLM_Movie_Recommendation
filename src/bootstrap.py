import logging

from src.candidate_retriever import FaissMovieCandidateRetriever
from src.chain import build_chain
from src.diversity import DiversityReranker
from src.embeddings import get_embedding_model, load_config
from src.hybrid_assistant import HybridMovieAssistant
from src.monitoring import init_mlflow
from src.recommender import MovieRecommender
from src.retriever import build_retriever, load_vector_store

logger = logging.getLogger(__name__)


def build_assistant() -> HybridMovieAssistant:
    """Assemble the RAG, MCP, FAISS candidate, and reranking pipeline."""

    config = load_config()
    init_mlflow(config)

    logger.info("Loading movie recommendation pipeline")

    # One embedding model is shared by RAG, FAISS retrieval, and reranking.
    embeddings = get_embedding_model(config)

    # Load FAISS only once.
    vector_store = load_vector_store(embeddings, config)

    # The LangChain retriever is used by the RAG question-answering chain.
    rag_retriever = build_retriever(vector_store, config)
    qa_chain = build_chain(rag_retriever, config)

    # Ranking configuration.
    ranking_config = config.get("ranking", {})
    ranking_weights = ranking_config.get("weights")
    candidate_config = ranking_config.get("candidates", {})
    diversity_config = ranking_config.get("diversity", {})

    # Reranks the merged TMDB and FAISS candidates.
    recommender = MovieRecommender(
        embeddings,
        weights=ranking_weights,
    )
    diversity_reranker = None
    if diversity_config.get("enabled", False):
        diversity_reranker = DiversityReranker(
            relevance_weight=diversity_config.get("relevance_weight", 0.85),
            max_per_collection=diversity_config.get("max_per_collection", 2),
        )

    # Uses the same FAISS index to discover additional movie candidates.
    candidate_retriever = FaissMovieCandidateRetriever(vector_store)

    assistant = HybridMovieAssistant(
        qa_chain=qa_chain,
        recommender=recommender,
        diversity_reranker=diversity_reranker,
        candidate_retriever=candidate_retriever,
        tmdb_candidate_limit=candidate_config.get("tmdb", 20),
        faiss_candidate_limit=candidate_config.get("faiss", 20),
        recommendation_limit=candidate_config.get("final", 5),
        diversity_candidate_pool=diversity_config.get("candidate_pool", 15),
    )

    logger.info("Movie recommendation pipeline ready")
    return assistant
