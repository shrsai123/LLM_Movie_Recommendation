from src.embeddings import PROJECT_ROOT, load_config


def test_load_config():
    config = load_config()
    assert isinstance(config, dict)
    assert config["model"]["embedding"] == "all-MiniLM-L6-v2"


def test_project_root():
    assert (PROJECT_ROOT / "config.yaml").exists()
    assert (PROJECT_ROOT / "src").is_dir()
