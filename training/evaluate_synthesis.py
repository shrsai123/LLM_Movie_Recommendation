"""Compare base Gemma and the synthesis adapter on identical holdout packets."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.response_synthesizer import ResponseSynthesizer


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_title(value: str) -> str:
    """Ignore capitalization and punctuation when matching movie headings."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _heading_titles(answer: str) -> list[str]:
    """Parse headings that follow `1. Movie Title (2020)` format."""
    titles = []
    pattern = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")
    for line in answer.splitlines():
        match = pattern.match(line)
        if match:
            # Strip the optional year after capturing the complete heading.
            title = re.sub(r"\s+\(\d{4}\)\s*$", "", match.group(1))
            titles.append(title.strip())
    return titles


def _token_f1(reference: str, prediction: str) -> float:
    """Calculate a lightweight overlap score without another ML dependency."""
    reference_tokens = Counter(re.findall(r"\b\w+\b", reference.casefold()))
    prediction_tokens = Counter(re.findall(r"\b\w+\b", prediction.casefold()))
    overlap = sum((reference_tokens & prediction_tokens).values())
    if not reference_tokens or not prediction_tokens or not overlap:
        return 0.0
    precision = overlap / sum(prediction_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _score_answer(record: dict, answer: str) -> dict:
    expected_titles = [
        str(movie["title"]) for movie in record["recommendations"]
    ]
    expected_normalized = [_normalize_title(title) for title in expected_titles]
    headings = _heading_titles(answer)
    heading_normalized = [_normalize_title(title) for title in headings]

    # Coverage gives partial credit when, for example, four of five movies are
    # mentioned. Order requires all expected movies to occur in the right order.
    lowered_answer = answer.casefold()
    positions = [lowered_answer.find(title.casefold()) for title in expected_titles]
    covered = sum(position >= 0 for position in positions)
    order_preserved = all(position >= 0 for position in positions) and positions == sorted(
        positions
    )

    unexpected = [
        title
        for title, normalized in zip(headings, heading_normalized, strict=True)
        if normalized not in expected_normalized
    ]
    format_valid = heading_normalized == expected_normalized

    return {
        "candidate_coverage": covered / len(expected_titles),
        "order_preserved": order_preserved,
        "format_valid": format_valid,
        # This is a deterministic hallucination proxy: an unexpected numbered
        # movie heading counts as an invented recommendation.
        "unexpected_heading_rate": len(unexpected) / max(1, len(headings)),
        "token_f1": _token_f1(record["reference_answer"], answer),
        "unexpected_headings": unexpected,
    }


def evaluate_model(synthesizer, records: list[dict]) -> dict:
    rows = []
    started = time.perf_counter()

    for record in records:
        try:
            # Validation is disabled only here so partial failures can still be
            # measured. Production inference keeps validation enabled.
            answer = synthesizer.generate(
                query=record["query"],
                source_movie=record.get("source_movie"),
                recommendations=record["recommendations"],
                validate_output=False,
            )
            scores = _score_answer(record, answer)
            error = None
        except Exception as exc:
            answer = ""
            scores = {
                "candidate_coverage": 0.0,
                "order_preserved": False,
                "format_valid": False,
                "unexpected_heading_rate": 0.0,
                "token_f1": 0.0,
                "unexpected_headings": [],
            }
            error = str(exc)

        rows.append(
            {
                "query": record["query"],
                "source_key": record.get("source_key"),
                "answer": answer,
                "error": error,
                **scores,
            }
        )

    elapsed = time.perf_counter() - started
    count = len(rows)
    return {
        "examples": count,
        "candidate_coverage": sum(row["candidate_coverage"] for row in rows) / count,
        "order_preservation_rate": sum(row["order_preserved"] for row in rows) / count,
        "format_valid_rate": sum(row["format_valid"] for row in rows) / count,
        "unexpected_heading_rate": sum(
            row["unexpected_heading_rate"] for row in rows
        )
        / count,
        "reference_token_f1": sum(row["token_f1"] for row in rows) / count,
        "generation_error_rate": sum(row["error"] is not None for row in rows) / count,
        "average_latency_seconds": elapsed / count,
        "examples_detail": rows,
    }


def clear_gpu_memory() -> None:
    """Release one model before loading the next comparison model."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _summary(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "examples_detail"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "synthesis" / "holdout.jsonl",
    )
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "recommendation-synthesis-adapter",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation" / "synthesis_comparison.json",
    )
    args = parser.parse_args()

    records = read_jsonl(args.data)
    if not records:
        raise ValueError("The holdout dataset is empty")

    print("Evaluating untouched base model...")
    base_model = ResponseSynthesizer(base_model=args.model, adapter_path=None)
    base_metrics = evaluate_model(base_model, records)
    del base_model
    clear_gpu_memory()

    print("Evaluating base model plus LoRA adapter...")
    fine_tuned_model = ResponseSynthesizer(
        base_model=args.model,
        adapter_path=str(args.adapter),
    )
    fine_tuned_metrics = evaluate_model(fine_tuned_model, records)
    del fine_tuned_model
    clear_gpu_memory()

    comparison = {
        "base_model": base_metrics,
        "fine_tuned_model": fine_tuned_metrics,
        "improvement": {
            key: _summary(fine_tuned_metrics)[key] - _summary(base_metrics)[key]
            for key in (
                "candidate_coverage",
                "order_preservation_rate",
                "format_valid_rate",
                "reference_token_f1",
            )
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "base_model": _summary(base_metrics),
        "fine_tuned_model": _summary(fine_tuned_metrics),
        "improvement": comparison["improvement"],
    }, indent=2))
    print(f"Saved detailed comparison to {args.out}")


if __name__ == "__main__":
    main()
