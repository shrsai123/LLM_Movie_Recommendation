import asyncio
import logging
import time

import gradio as gr

from src.chain import build_chain
from src.embeddings import get_embedding_model, load_config
from src.hybrid_assistant import HybridMovieAssistant
from src.monitoring import init_mlflow, track_query
from src.retriever import load_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize lightweight app config. The RAG pipeline is loaded lazily so the
# Gradio UI and TMDB-only tools can run before the full LLM finishes loading.
logger.info("Starting Movie Recommender")
config = load_config()
init_mlflow(config)

assistant = None


def build_qa_chain():
    start = time.perf_counter()
    logger.info("Loading RAG pipeline")

    embeddings = get_embedding_model(config)
    logger.info("Embedding model ready in %.2fs", time.perf_counter() - start)

    retriever_start = time.perf_counter()
    retriever = load_retriever(embeddings, config)
    logger.info("FAISS retriever ready in %.2fs", time.perf_counter() - retriever_start)

    chain_start = time.perf_counter()
    qa = build_chain(retriever, config)
    logger.info("LLM chain ready in %.2fs", time.perf_counter() - chain_start)
    logger.info("RAG pipeline ready in %.2fs", time.perf_counter() - start)
    return qa


async def load_qa_chain():
    return await asyncio.to_thread(build_qa_chain)


assistant = HybridMovieAssistant(qa_chain_loader=load_qa_chain)


"""@track_query
def handle_conversation(message, history):
    result = qa.invoke({"query": message})
    response = result["result"]

    # Strip prompt leakage
    if "Your response:" in response:
        response = response.split("Your response:")[-1].strip()

    return response"""


@track_query
async def handle_conversation(message, history):
    result = await assistant.answer(message)
    return result["answer"]


demo = gr.ChatInterface(
    fn=handle_conversation,
    title="Movie Blasters",
    description="Your AI-powered movie recommendation assistant",
    chatbot=gr.Chatbot(
        value=[],
        height="calc(100vh - 200px)",
        container=True,
    ),
)

if __name__ == "__main__":
    demo.launch(
        server_name=config.get("app", {}).get("host", "0.0.0.0"),
        server_port=config.get("app", {}).get("port", 7860),
        inbrowser=True,
        show_error=True,
    )
