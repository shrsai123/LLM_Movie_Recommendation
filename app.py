import logging
from src.embeddings import get_embedding_model, load_config
from src.retriever import load_retriever
from src.chain import build_chain
import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize pipeline
logger.info("Starting Movie Recommender")
config = load_config()
embeddings = get_embedding_model(config)
retriever = load_retriever(embeddings, config)
qa = build_chain(retriever, config)
logger.info("Pipeline ready")


def handle_conversation(message, history):
    result = qa.invoke({"query": message})
    response = result["result"]

    # Strip prompt leakage
    if "Your response:" in response:
        response = response.split("Your response:")[-1].strip()

    return response


demo = gr.ChatInterface(
    fn=handle_conversation,
    title="Movie Blasters",
    description="Your AI-powered movie recommendation assistant",
    chatbot=gr.Chatbot(
        value=[],
        height="calc(100vh - 200px)",
        container=True,
    )

)

if __name__ == "__main__":
    demo.launch(
        server_name=config.get("app", {}).get("host", "0.0.0.0"),
        server_port=config.get("app", {}).get("port", 7860),
    )