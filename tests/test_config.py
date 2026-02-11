from pathlib import Path

import yaml


def test_config_exists():
    assert Path("config.yaml").exists()


def test_config_has_required_keys():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    assert "model" in config
    assert "embedding" in config["model"]
    assert "llm" in config["model"]
    assert "retrieval" in config
    assert "top_k" in config["retrieval"]
    assert "mlflow" in config
