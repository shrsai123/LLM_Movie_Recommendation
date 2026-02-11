# src/monitoring.py
import logging
import os
import time
from functools import wraps

logger = logging.getLogger(__name__)

MLFLOW_AVAILABLE = False


def init_mlflow(config: dict):
    """Initialize MLflow connection. Fail fast if unavailable."""
    global MLFLOW_AVAILABLE

    # Skip if explicitly disabled
    if os.environ.get("MLFLOW_DISABLED", "false").lower() == "true":
        logger.info("MLflow disabled via environment variable")
        MLFLOW_AVAILABLE = False
        return

    try:
        # Quick check if server is reachable before importing mlflow
        import urllib.request

        tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
        urllib.request.urlopen(tracking_uri, timeout=3)

        import mlflow

        experiment_name = config.get("mlflow", {}).get("experiment_name", "movie-recommender")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        MLFLOW_AVAILABLE = True
        logger.info(f"MLflow tracking at {tracking_uri}, experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"MLflow unavailable, running without tracking: {e}")
        MLFLOW_AVAILABLE = False


def track_query(func):
    """Decorator to log each query to MLflow. Skips if MLflow is down."""

    @wraps(func)
    def wrapper(message, history):
        start = time.perf_counter()
        response = func(message, history)
        latency = time.perf_counter() - start

        if MLFLOW_AVAILABLE:
            try:
                import mlflow

                with mlflow.start_run(nested=True):
                    mlflow.log_param("query", message[:200])
                    mlflow.log_metric("latency_seconds", latency)
                    mlflow.log_metric("response_length", len(response))
                    mlflow.log_metric("history_length", len(history) if history else 0)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        return response

    return wrapper
