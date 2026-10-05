"""Build synthesis SFT data directly from a source-movie list.

Updated self-contained version: candidates are fetched directly through MCP.

For each source movie, this script asks the TMDB MCP server for candidates,
applies the existing four-signal recommender and diversity reranker, then
writes train/dev/holdout JSONL files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.diversity import DiversityReranker
from src.embeddings import get_embedding_model, load_config
from src.mcp_client import MovieMCPClient
from src.recommender import MovieRecommender
from src.response_synthesizer import build_messages


QUERY_TEMPLATES = (
    "Give me movies like {title}",
    "What should I watch if I liked {title}?",
    "Recommend films similar to {title}",
)


def _first_sentence(text: str, maximum: int = 260) -> str:
    """Keep generated training answers short and grounded."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0]
    return sentence[:maximum].rstrip()


def _movie_label(movie: dict) -> str:
    title = movie.get("title", "Unknown title")
    year = movie.get("release_year")
    return f"{title} ({year})" if year else title


def _reference_completion(source: dict, recommendations: list[dict]) -> str:
    """Create a deterministic draft response for supervised fine-tuning."""
    source_label = _movie_label(source)
    sections = [f"If you enjoyed {source_label}, these are strong options:"]

    for rank, movie in enumerate(recommendations, start=1):
        reason = str(movie.get("reason") or "").strip()
        overview = _first_sentence(str(movie.get("overview") or ""))
        explanation_parts = [
            reason.rstrip(".")
            if reason
            else f"This movie was selected as a strong match for {source_label}"
        ]
        if overview:
            explanation_parts.append(overview.rstrip("."))

        explanation = ". ".join(explanation_parts) + "."
        sections.append(f"{rank}. {_movie_label(movie)}\n{explanation}")

    return "\n\n".join(sections)


def _load_source_definitions(path: Path) -> list[dict]:
    """Load and validate the source-movie manifest."""
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or len(items) < 3:
        raise ValueError("The source file must contain at least three movies")

    seen = set()
    validated = []
    for item in items:
        title = str(item.get("title") or "").strip()
        if not title:
            raise ValueError("Every source entry requires a title")

        year = item.get("year")
        key = (title.casefold(), year)
        if key in seen:
            raise ValueError(f"Duplicate source movie: {title} ({year})")
        seen.add(key)

        split = item.get("split", "development")
        if split not in {"development", "holdout"}:
            raise ValueError(
                f"{title}: split must be either 'development' or 'holdout'"
            )
        validated.append({"title": title, "year": year, "split": split})
    return validated


def _assign_source_splits(packets: list[dict], seed: int) -> dict[str, str]:
    """Split by source movie so query variants never leak between sets."""
    randomizer = random.Random(seed)
    holdout = [item for item in packets if item["split_hint"] == "holdout"]
    development = [item for item in packets if item["split_hint"] != "holdout"]
    randomizer.shuffle(development)

    if not holdout:
        holdout_count = max(1, round(len(development) * 0.20))
        holdout = development[-holdout_count:]
        development = development[:-holdout_count]

    if len(development) < 2:
        raise ValueError("Use at least two non-holdout source movies")

    dev_count = max(1, round(len(development) * 0.20))
    validation = development[-dev_count:]
    training = development[:-dev_count]
    if not training:
        raise ValueError("Not enough development movies to create train and dev sets")

    assignments = {item["source_key"]: "train" for item in training}
    assignments.update({item["source_key"]: "dev" for item in validation})
    assignments.update({item["source_key"]: "holdout" for item in holdout})
    return assignments


def _record(packet: dict, query: str) -> dict:
    source = packet["source_movie"]
    recommendations = packet["recommendations"]
    completion = _reference_completion(source, recommendations)
    return {
        "prompt": build_messages(query, recommendations, source_movie=source),
        "completion": [{"role": "assistant", "content": completion}],
        "query": query,
        "source_movie": source,
        "recommendations": recommendations,
        "reference_answer": completion,
        "source_key": packet["source_key"],
    }


async def _build_packet(
    source_definition: dict,
    mcp: MovieMCPClient,
    recommender: MovieRecommender,
    diversity_reranker: DiversityReranker | None,
    tmdb_limit: int,
    candidate_pool: int,
    final_limit: int,
) -> dict:
    """Fetch, rank, and diversify recommendations for one source movie."""
    arguments = {"title": source_definition["title"], "limit": tmdb_limit}
    if source_definition.get("year") is not None:
        arguments["year"] = source_definition["year"]

    data = await mcp.call_tool("get_similar_movies", arguments)
    if not isinstance(data, dict) or data.get("error"):
        detail = data.get("error") if isinstance(data, dict) else "invalid MCP response"
        raise RuntimeError(f"{source_definition['title']}: {detail}")

    source_movie = data.get("source_movie") or {}
    candidates = data.get("recommendations") or []
    if len(candidates) < final_limit:
        raise ValueError(
            f"{source_definition['title']}: TMDB returned only {len(candidates)} "
            f"candidates; {final_limit} are required"
        )

    # Four relevance signals: overview, keywords, genres, and collection.
    relevance_limit = min(len(candidates), max(candidate_pool, final_limit))
    relevance_ranked = recommender.rank(
        source_movie,
        candidates,
        limit=relevance_limit,
    )

    # Final pass reduces redundant results while retaining relevance.
    recommendations = (
        diversity_reranker.rerank(relevance_ranked, limit=final_limit)
        if diversity_reranker is not None
        else relevance_ranked[:final_limit]
    )

    return {
        "source_key": str(source_movie.get("id") or source_definition["title"]),
        "split_hint": source_definition["split"],
        "source_movie": source_movie,
        "recommendations": recommendations,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def build_dataset(args: argparse.Namespace) -> None:
    """Run collection, ranking, splitting, and JSONL generation."""
    config = load_config()
    ranking_config = config.get("ranking", {})
    candidate_config = ranking_config.get("candidates", {})
    diversity_config = ranking_config.get("diversity", {})

    tmdb_limit = args.tmdb_limit or int(candidate_config.get("tmdb", 20))
    final_limit = args.limit or int(candidate_config.get("final", 5))
    candidate_pool = int(diversity_config.get("candidate_pool", 15))
    if tmdb_limit < final_limit:
        raise ValueError("TMDB candidate limit cannot be smaller than final limit")

    source_definitions = _load_source_definitions(args.sources)
    embeddings = get_embedding_model(config)
    recommender = MovieRecommender(
        embeddings,
        weights=ranking_config.get("weights"),
    )

    diversity_reranker = None
    if diversity_config.get("enabled", True):
        diversity_reranker = DiversityReranker(
            relevance_weight=float(diversity_config.get("relevance_weight", 0.85)),
            max_per_collection=int(diversity_config.get("max_per_collection", 2)),
        )

    mcp = MovieMCPClient()
    packets = []
    for position, source_definition in enumerate(source_definitions, start=1):
        print(
            f"[{position}/{len(source_definitions)}] Building recommendations for "
            f"{source_definition['title']}"
        )
        packets.append(
            await _build_packet(
                source_definition,
                mcp,
                recommender,
                diversity_reranker,
                tmdb_limit,
                candidate_pool,
                final_limit,
            )
        )

    assignments = _assign_source_splits(packets, args.seed)
    splits = {"train": [], "dev": [], "holdout": []}
    for packet in packets:
        source_title = packet["source_movie"]["title"]
        split = assignments[packet["source_key"]]
        for template in QUERY_TEMPLATES:
            query = template.format(title=source_title)
            splits[split].append(_record(packet, query))

    randomizer = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        randomizer.shuffle(records)
        _write_jsonl(args.out / f"{split}.jsonl", records)
        source_count = len({record["source_key"] for record in records})
        print(f"{split}: {len(records)} examples from {source_count} sources")

    print(f"Dataset written to {args.out}")
    print("Review and improve reference_answer values before training.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation" / "sources.json",
        help="JSON list containing title, year, and development/holdout split",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "synthesis",
    )
    parser.add_argument("--tmdb-limit", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    asyncio.run(build_dataset(args))


if __name__ == "__main__":
    main()
