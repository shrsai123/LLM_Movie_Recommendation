import logging

from src.candidate_retriever import FaissMovieCandidateRetriever
from src.diversity import DiversityReranker
from src.embeddings import get_embedding_model, load_config
from src.hybrid_assistant import HybridMovieAssistant
from src.monitoring import init_mlflow
from src.preference_recommender import PreferenceRecommender
from src.recommender import MovieRecommender
from src.response_synthesizer import ResponseSynthesizer, SynthesisError
from src.retriever import load_vector_store
from src.workflow import LangGraphMovieAssistant

logger = logging.getLogger(__name__)


def _build_response_synthesizer(config):
    synthesis_config = config.get("post_training", {}).get("synthesis", {})

    if not synthesis_config.get("enabled", False):
        return None

    try:
        return ResponseSynthesizer(
            base_model=synthesis_config.get(
                "base_model",
                "google/gemma-3-1b-it",
            ),
            adapter_path=synthesis_config.get(
                "adapter_path",
                "artifacts/recommendation-synthesis-adapter",
            ),
            quantize_4bit=synthesis_config.get("quantize_4bit", True),
            max_new_tokens=synthesis_config.get("max_new_tokens", 250),
            device=synthesis_config.get("device", "auto"),
        )
    except (SynthesisError, OSError, ValueError):
        logger.exception(
            "Fine-tuned synthesis model could not be loaded; "
            "using deterministic formatting"
        )
        return None


def build_assistant() -> HybridMovieAssistant:
    """Assemble the MCP, FAISS candidate, and reranking pipeline."""

    config = load_config()
    init_mlflow(config)

    logger.info("Loading movie recommendation pipeline")

    # One embedding model is shared by FAISS candidate retrieval and reranking.
    embeddings = get_embedding_model(config)

    # Load FAISS only once.
    vector_store = load_vector_store(embeddings, config)

    # Ranking configuration.
    ranking_config = config.get("ranking", {})
    ranking_weights = ranking_config.get("weights")
    preference_weights = config.get("ranking", {}).get("preference_weights")
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
        recommender=recommender,
        preference_recommender=PreferenceRecommender(
            embeddings,
            weights=preference_weights,
        ),
        diversity_reranker=diversity_reranker,
        candidate_retriever=candidate_retriever,
        tmdb_candidate_limit=candidate_config.get("tmdb", 20),
        faiss_candidate_limit=candidate_config.get("faiss", 20),
        recommendation_limit=candidate_config.get("final", 5),
        diversity_candidate_pool=diversity_config.get("candidate_pool", 15),
        response_synthesizer=_build_response_synthesizer(config),
    )

    logger.info("Movie recommendation pipeline ready")
    return LangGraphMovieAssistant(assistant)
