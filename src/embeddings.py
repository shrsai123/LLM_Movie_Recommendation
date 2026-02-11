from pathlib import Path

import yaml
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


"""def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)"""


def get_embedding_model(config):
    return HuggingFaceEmbeddings(model_name=config["model"]["embedding"])
