import logging
import asyncio

import gradio as gr

from src.chain import build_chain
from src.embeddings import get_embedding_model, load_config
from src.monitoring import init_mlflow, track_query
from src.retriever import load_retriever
from src.hybrid_assistant import HybridMovieAssistant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize lightweight app config. The model pipeline is loaded lazily so the
# Gradio UI can come up before the full LLM finishes loading.
logger.info("Starting Movie Recommender")
config = load_config()
init_mlflow(config)

assistant = None
assistant_lock = asyncio.Lock()


def build_assistant():
    logger.info("Loading recommendation pipeline")
    embeddings = get_embedding_model(config)
    retriever = load_retriever(embeddings, config)
    qa = build_chain(retriever, config)
    logger.info("Pipeline ready")
    return HybridMovieAssistant(qa)


async def get_assistant():
    global assistant

    if assistant is None:
        async with assistant_lock:
            if assistant is None:
                assistant = await asyncio.to_thread(build_assistant)

    return assistant


'''@track_query
def handle_conversation(message, history):
    result = qa.invoke({"query": message})
    response = result["result"]

    # Strip prompt leakage
    if "Your response:" in response:
        response = response.split("Your response:")[-1].strip()

    return response'''

@track_query
async def handle_conversation(message, history):
    active_assistant = await get_assistant()
    result = await active_assistant.answer(message)
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
