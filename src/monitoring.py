# src/monitoring.py
import logging
import time
from functools import wraps

import mlflow

logger = logging.getLogger(__name__)


def init_mlflow(config: dict):
    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "http://localhost:5000")
    experiment_name = config.get("mlflow", {}).get("experiment_name", "movie-recommender")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking at {tracking_uri}, experiment: {experiment_name}")


def track_query(func):
    @wraps(func)
    def wrapper(message, history):
        start = time.perf_counter()
        response = func(message, history)
        latency = time.perf_counter() - start

        try:
            with mlflow.start_run(nested=True):
                mlflow.log_param("query", message[:200])
                mlflow.log_metric("latency_seconds", latency)
                mlflow.log_metric("response_length", len(response))
                mlflow.log_metric("history_length", len(history) if history else 0)
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

        return response

    return wrapper
