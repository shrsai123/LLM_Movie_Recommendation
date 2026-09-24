"""Prepare one shared candidate set and compare TMDB against local rankings."""

import argparse
import asyncio
import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.diversity import DiversityReranker

PROMPT_VERSION = "movie-relevance-v2"
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "integer", "enum": [0, 1, 2]},
        "reason": {"type": "string", "minLength": 1},
    },
    "required": ["relevance", "reason"],
    "additionalProperties": False,
}
GENRE_NAMES = {
    12: "Adventure", 14: "Fantasy", 16: "Animation", 18: "Drama", 27: "Horror",
    28: "Action", 35: "Comedy", 36: "History", 37: "Western", 53: "Thriller",
    80: "Crime", 99: "Documentary", 878: "Science Fiction", 9648: "Mystery",
    10402: "Music", 10749: "Romance", 10751: "Family", 10752: "War",
    10759: "Action & Adventure", 10762: "Kids", 10763: "News", 10764: "Reality",
    10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk", 10768: "War & Politics",
    10769: "Foreign", 10770: "TV Movie",
}


def genre_names(movie: dict) -> list[str]:
    return [GENRE_NAMES.get(genre_id, str(genre_id)) for genre_id in movie.get("genre_ids") or []]


def movie_for_judge(movie: dict) -> dict:
    """Expose content features only; ranking metadata must never reach the judge."""
    return {
        "title": movie.get("title", ""),
        "overview": movie.get("overview", ""),
        "genres": genre_names(movie),
    }


def judge_prompt(source: dict, candidate: dict) -> str:
    return f"""You are evaluating whether one movie is a meaningful recommendation for a viewer who liked another movie.

Rate relevance using this rubric:
- 0: little meaningful similarity
- 1: shares a genre, theme, setting, or tone
- 2: strongly similar across multiple important attributes

Consider plot, themes, setting, tone, and genres. Do not mention ranking or scores.

Source movie:
{json.dumps(movie_for_judge(source), ensure_ascii=False)}

Candidate movie:
{json.dumps(movie_for_judge(candidate), ensure_ascii=False)}

Return exactly one line of JSON. Your first character must be `{{` and your last character must be `}}`:
{{"relevance": 0, "reason": "brief content-based reason"}}
"""


def retry_prompt(source: dict, candidate: dict) -> str:
    return judge_prompt(source, candidate) + "\nJSON only. Do not explain your reasoning before the JSON."


def parse_judgment(text: str) -> dict:
    """Extract and validate the first JSON object returned by the judge model."""
    decoder = json.JSONDecoder()
    for start in (index for index, char in enumerate(text) if char == "{"):
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if payload.get("relevance") in {0, 1, 2} and isinstance(payload.get("reason"), str):
            return {"relevance": payload["relevance"], "reason": payload["reason"].strip()}
    raise ValueError(f"Judge did not return valid relevance JSON: {text!r}")


def ndcg_at_k(order: list[int], labels: dict[int, int], k: int = 5) -> float | None:
    """Use graded gains (2**relevance - 1); return None if nothing is relevant."""

    def dcg(ratings: list[int]) -> float:
        return sum((2**rating - 1) / math.log2(rank + 2) for rank, rating in enumerate(ratings))

    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    if ideal == 0:
        return None
    return dcg([labels[movie_id] for movie_id in order[:k]]) / ideal


def deduplicate(source_id: int, candidates: list[dict]) -> list[dict]:
    """Match MovieRecommender's source and repeated-ID removal."""
    seen = {source_id}
    unique = []
    for movie in candidates:
        movie_id = movie.get("id")
        if movie_id is None:
            raise ValueError("Evaluation requires TMDB IDs for every candidate")
        if movie_id not in seen:
            seen.add(movie_id)
            unique.append(movie)
    return unique


def diversity_from_config(config: dict) -> tuple[DiversityReranker | None, int | None]:
    diversity_config = config.get("ranking", {}).get("diversity", {})
    if not diversity_config.get("enabled", False):
        return None, None
    return (
        DiversityReranker(
            relevance_weight=diversity_config.get("relevance_weight", 0.85),
            max_per_collection=diversity_config.get("max_per_collection", 2),
        ),
        diversity_config.get("candidate_pool", 15),
    )


def rank_with_diversity(
    source: dict,
    candidates: list[dict],
    recommender,
    diversity_reranker: DiversityReranker | None = None,
    candidate_pool: int | None = None,
) -> list[dict]:
    """Rank every candidate while diversifying the production-sized head."""
    ranked = recommender.rank(source, candidates, len(candidates))
    if diversity_reranker is None:
        return ranked

    pool_size = min(candidate_pool or len(ranked), len(ranked))
    diversified = diversity_reranker.rerank(ranked[:pool_size], pool_size)
    return diversified + ranked[pool_size:]


async def prepare(sources_path: Path, output_dir: Path) -> None:
    # Keep model and MCP imports here so `score` works without either dependency.
    from src.embeddings import get_embedding_model, load_config
    from src.mcp_client import MovieMCPClient
    from src.recommender import MovieRecommender

    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    if not 10 <= len(sources) <= 20:
        raise ValueError("Choose 10–20 source movies in sources.json")
    if any(entry.get("split") not in {"development", "holdout"} for entry in sources):
        raise ValueError("Each source needs split: development or holdout")

    config = load_config()
    recommender = MovieRecommender(
        get_embedding_model(config),
        weights=config.get("ranking", {}).get("weights"),
    )
    diversity_reranker, candidate_pool = diversity_from_config(config)
    mcp = MovieMCPClient()
    snapshot = []
    label_rows = []

    for entry in sources:
        data = await mcp.call_tool(
            "get_similar_movies",
            {"title": entry["title"], "year": entry["year"], "limit": 20},
        )
        if data.get("error"):
            raise ValueError(f"{entry['title']}: {data['error']}")
        source = data["source_movie"]
        if source.get("id") is None:
            raise ValueError(f"No TMDB ID for source: {entry['title']}")
        candidates = deduplicate(source["id"], data.get("recommendations") or [])
        if not candidates:
            raise ValueError(f"No candidates for {entry['title']}")
        ranked = await asyncio.to_thread(
            rank_with_diversity,
            source,
            candidates,
            recommender,
            diversity_reranker,
            candidate_pool,
        )
        key = f"{source['id']}"
        snapshot.append(
            {
                "source_id": key,
                "source_title": source["title"],
                "split": entry["split"],
                "source_movie": source,
                "candidates": candidates,
                "tmdb_ids": [movie["id"] for movie in candidates],
                "reranked_ids": [movie["id"] for movie in ranked],
            }
        )
        for movie in candidates:
            label_rows.append(
                {
                    "source_id": key,
                    "source_title": source["title"],
                    "source_year": source.get("release_year") or "",
                    "candidate_id": movie["id"],
                    "candidate_title": movie["title"],
                    "candidate_year": movie.get("release_year") or "",
                    "candidate_overview": movie.get("overview") or "",
                    "relevance": "",
                    "reason": "",
                    "label_source": "",
                    "judge_model": "",
                    "prompt_version": "",
                    "second_pass_relevance": "",
                    "needs_review": "",
                }
            )
        print(f"{source['title']}: {len(candidates)} unique candidates")

    # Shuffle within each source so the labeling sheet does not show TMDB's order.
    random.Random(42).shuffle(label_rows)
    output_dir.mkdir(parents=True)
    (output_dir / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)
    print(f"Rate 0, 1, or 2 in {output_dir / 'labels.csv'}; keep snapshot.json unchanged.")


@dataclass
class GeminiJudge:
    client: object
    api_key: str
    model_id: str


class GeminiRequestError(RuntimeError):
    pass


def load_judge(model_id: str, provider: str):
    if provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Set GEMINI_API_KEY before using --provider gemini")
        import httpx

        return GeminiJudge(
            client=httpx.Client(timeout=60),
            api_key=os.environ["GEMINI_API_KEY"],
            model_id=model_id,
        )

    from transformers import pipeline

    judge = pipeline("text-generation", model=model_id, device_map="auto")
    # Sampling fields bundled with some model configs are irrelevant for greedy judging.
    for setting in ("temperature", "top_p", "top_k"):
        setattr(judge.model.generation_config, setting, None)
    return judge


def run_judge(judge, source: dict, candidate: dict) -> dict:
    if isinstance(judge, GeminiJudge):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{judge.model_id}:generateContent"
        headers = {"x-goog-api-key": judge.api_key}
        contents = [{"parts": [{"text": judge_prompt(source, candidate)}]}]
        response = judge.client.post(
            url,
            headers=headers,
            json={
                "contents": contents,
                "generationConfig": {
                    "responseFormat": {
                        "text": {"mimeType": "application/json", "schema": JUDGMENT_SCHEMA}
                    }
                },
            },
        )
        if response.status_code == 400:
            # Older Gemini API deployments use these equivalent structured-output fields.
            response = judge.client.post(
                url,
                headers=headers,
                json={
                    "contents": contents,
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseJsonSchema": JUDGMENT_SCHEMA,
                    },
                },
            )
        if response.is_error:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise GeminiRequestError(f"Gemini API request failed ({response.status_code}): {detail}")
        payload = response.json()
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"Gemini returned no text response: {payload}") from exc
        return parse_judgment(text)

    def generate(prompt: str, max_new_tokens: int) -> dict:
        response = judge(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return parse_judgment(response[0]["generated_text"])

    try:
        return generate(judge_prompt(source, candidate), max_new_tokens=96)
    except ValueError:
        return generate(retry_prompt(source, candidate), max_new_tokens=96)


def ensure_labels_writable(labels_path: Path) -> None:
    try:
        with labels_path.open("a", encoding="utf-8-sig"):
            pass
    except PermissionError as exc:
        raise ValueError(
            f"Cannot write {labels_path}. Close it in Excel or another CSV viewer before labeling."
        ) from exc


def write_labels(labels_path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    try:
        with labels_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError as exc:
        recovery_path = labels_path.with_name(f"{labels_path.stem}.pending{labels_path.suffix}")
        with recovery_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        raise ValueError(
            f"Cannot write {labels_path}; results were saved to {recovery_path}. "
            "Close the original CSV, then replace it with the recovery file."
        ) from exc


def auto_label(
    run_dir: Path,
    model_id: str,
    passes: int,
    refresh_llm: bool,
    provider: str = "gemini",
) -> None:
    if passes not in {1, 2}:
        raise ValueError("--passes must be 1 or 2")

    snapshot = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
    movies = {
        (str(item["source_id"]), int(candidate["id"])): (item["source_movie"], candidate)
        for item in snapshot
        for candidate in item["candidates"]
    }
    labels_path = run_dir / "labels.csv"
    ensure_labels_writable(labels_path)
    with labels_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required_fields = [
        "reason", "label_source", "judge_model", "prompt_version", "second_pass_relevance", "needs_review",
    ]
    fieldnames = list(dict.fromkeys(fieldnames + required_fields))
    judge = load_judge(model_id, provider)
    eligible = [
        row for row in rows
        if not row["relevance"].strip() or (refresh_llm and row.get("label_source") == "llm")
    ]
    random.Random(42).shuffle(eligible)

    labeled = 0
    invalid = 0
    for row in eligible:
        key = (row["source_id"], int(row["candidate_id"]))
        try:
            source, candidate = movies[key]
        except KeyError as exc:
            raise ValueError(f"Label does not match snapshot: {key[0]} / {key[1]}") from exc

        try:
            first = run_judge(judge, source, candidate)
        except ValueError:
            row["relevance"] = ""
            row["reason"] = ""
            row["label_source"] = "llm"
            row["judge_model"] = model_id
            row["prompt_version"] = PROMPT_VERSION
            row["second_pass_relevance"] = ""
            row["needs_review"] = "invalid_judge_output"
            invalid += 1
            continue
        row["relevance"] = str(first["relevance"])
        row["reason"] = first["reason"]
        row["label_source"] = "llm"
        row["judge_model"] = model_id
        row["prompt_version"] = PROMPT_VERSION
        row["second_pass_relevance"] = ""
        row["needs_review"] = ""
        if passes == 2:
            second = run_judge(judge, source, candidate)
            row["second_pass_relevance"] = str(second["relevance"])
            if second["relevance"] != first["relevance"]:
                row["needs_review"] = "inconsistent_judge"
        labeled += 1

    write_labels(labels_path, fieldnames, rows)
    (run_dir / "labeling_metadata.json").write_text(
        json.dumps(
            {
                "judge_model": model_id,
                "provider": provider,
                "prompt_version": PROMPT_VERSION,
                "passes": passes,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"LLM-labeled {labeled} rows; {invalid} rows need review after invalid judge output. "
        "Existing manual labels were preserved."
    )


def score(run_dir: Path, snapshot_path: Path | None = None) -> list[dict]:
    snapshot_path = snapshot_path or run_dir / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    labels: dict[str, dict[int, int]] = defaultdict(dict)
    with (run_dir / "labels.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            source_id, movie_id = row["source_id"], int(row["candidate_id"])
            rating = row["relevance"].strip()
            if rating not in {"0", "1", "2"}:
                raise ValueError(f"Rate every row 0, 1, or 2: {source_id} / {movie_id}")
            if movie_id in labels[source_id]:
                raise ValueError(f"Duplicate label for {source_id} / {movie_id}")
            labels[source_id][movie_id] = int(rating)

    results = []
    for item in snapshot:
        source_id = item["source_id"]
        tmdb_ids, reranked_ids = item["tmdb_ids"], item["reranked_ids"]
        if (
            len(tmdb_ids) != len(set(tmdb_ids))
            or len(tmdb_ids) != len(reranked_ids)
            or set(tmdb_ids) != set(reranked_ids)
        ):
            raise ValueError(f"Candidate sets differ for {item['source_title']}")
        if set(tmdb_ids) != set(labels[source_id]):
            raise ValueError(f"Missing or extra labels for {item['source_title']}")
        baseline = ndcg_at_k(tmdb_ids, labels[source_id])
        reranked = ndcg_at_k(reranked_ids, labels[source_id])
        if baseline is None:
            print(f"{item['source_title']}: no relevant candidates; excluded from average")
            continue
        result = {
            "source": item["source_title"],
            "split": item["split"],
            "tmdb": baseline,
            "reranked": reranked,
            "delta": reranked - baseline,
        }
        results.append(result)
        print(
            f"{result['source']} ({result['split']}): TMDB {baseline:.3f} | "
            f"reranked {reranked:.3f} | difference {result['delta']:+.3f}"
        )

    for split in ("development", "holdout"):
        rows = [result for result in results if result["split"] == split]
        if rows:
            print(
                f"{split} average ({len(rows)} sources): "
                f"TMDB {sum(row['tmdb'] for row in rows) / len(rows):.3f} | "
                f"reranked {sum(row['reranked'] for row in rows) / len(rows):.3f}"
            )
    return results


def rerank_snapshot(
    snapshot: list[dict],
    recommender,
    diversity_reranker: DiversityReranker | None = None,
    candidate_pool: int | None = None,
) -> list[dict]:
    """Recompute rankings while preserving the frozen source and candidate pool."""
    reranked_snapshot = []
    for item in snapshot:
        source = item["source_movie"]
        candidates = deduplicate(source["id"], item["candidates"])
        ranked = rank_with_diversity(
            source,
            candidates,
            recommender,
            diversity_reranker,
            candidate_pool,
        )
        reranked_snapshot.append(
            {
                **item,
                "previous_reranked_ids": item.get("reranked_ids", []),
                "reranked_ids": [movie["id"] for movie in ranked],
            }
        )
    return reranked_snapshot


def rerank(
    run_dir: Path,
    output_path: Path | None = None,
    snapshot_path: Path | None = None,
) -> Path:
    from src.embeddings import get_embedding_model, load_config
    from src.recommender import MovieRecommender

    snapshot_path = snapshot_path or run_dir / "snapshot_four_signals.json"
    output_path = output_path or run_dir / "snapshot_semantic_keywords.json"
    if output_path.exists():
        raise ValueError(f"Output file already exists: {output_path}")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    config = load_config()
    recommender = MovieRecommender(
        get_embedding_model(config),
        weights=config.get("ranking", {}).get("weights"),
    )
    diversity_reranker, candidate_pool = diversity_from_config(config)
    reranked = rerank_snapshot(
        snapshot,
        recommender,
        diversity_reranker,
        candidate_pool,
    )
    output_path.write_text(
        json.dumps(reranked, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote refreshed rankings to {output_path}")
    return output_path


async def enrich_snapshot(run_dir: Path, output_path: Path | None = None) -> Path:
    """Add four-signal metadata without changing the frozen candidate pool or labels."""
    from mcp_server.tmdb_client import TMDBClient
    from src.embeddings import get_embedding_model, load_config
    from src.recommender import MovieRecommender

    source_path = run_dir / "snapshot.json"
    snapshot = json.loads(source_path.read_text(encoding="utf-8"))
    output_path = output_path or run_dir / "snapshot_four_signals.json"

    if output_path.exists():
        raise ValueError(f"Output file already exists: {output_path}")

    config = load_config()
    recommender = MovieRecommender(
        get_embedding_model(config),
        weights=config.get("ranking", {}).get("weights"),
    )
    diversity_reranker, candidate_pool = diversity_from_config(config)
    tmdb = TMDBClient()
    enriched_snapshot = []

    for item in snapshot:
        movies = [item["source_movie"], *item["candidates"]]
        enriched = await tmdb.enrich_movies(movies)
        source, candidates = enriched[0], enriched[1:]
        ranked = await asyncio.to_thread(
            rank_with_diversity,
            source,
            candidates,
            recommender,
            diversity_reranker,
            candidate_pool,
        )

        enriched_snapshot.append(
            {
                **item,
                "source_movie": source,
                "candidates": candidates,
                "previous_reranked_ids": item.get("reranked_ids", []),
                "reranked_ids": [movie["id"] for movie in ranked],
                "ranking_signals": [
                    "overview",
                    "keywords",
                    "genres",
                    "collection",
                ],
            }
        )
        print(f"{item['source_title']}: enriched {len(candidates)} frozen candidates")

    output_path.write_text(
        json.dumps(enriched_snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote four-signal snapshot to {output_path}")
    return output_path


def candidate_scores(
    source: dict,
    candidates: list[dict],
    recommender,
) -> dict[int, dict[str, float | None]]:
    """Use the production recommender so diagnostic scoring cannot drift."""
    ranked = recommender.rank(source, candidates, limit=len(candidates))
    return {
        int(movie["id"]): {
            "overview_similarity": movie.get("signals", {}).get("overview"),
            "keyword_similarity": movie.get("signals", {}).get("keywords"),
            "genre_similarity": movie.get("signals", {}).get("genres"),
            "collection_match": movie.get("signals", {}).get("collection"),
            "final_score": movie.get("score", 0.0),
        }
        for movie in ranked
    }


def diagnostic_issue(relevance: int, tmdb_rank: int, reranker_rank: int, k: int = 5) -> str:
    if relevance <= 1 and reranker_rank <= k and tmdb_rank > k:
        return "bad_promotion"
    if relevance == 2 and tmdb_rank <= k and reranker_rank > k:
        return "bad_demotion"
    if relevance <= 1 and reranker_rank <= k:
        return "low_relevance_top5"
    if relevance == 2 and reranker_rank > k:
        return "high_relevance_below_top5"
    return ""


def build_diagnostic_rows(
    snapshot: list[dict],
    label_rows: list[dict],
    recommender,
    only_regressions: bool = False,
) -> list[dict]:
    labels_by_pair = {
        (str(row["source_id"]), int(row["candidate_id"])): row for row in label_rows
    }
    output = []

    for item in snapshot:
        source_id = str(item["source_id"])
        tmdb_ids = [int(movie_id) for movie_id in item["tmdb_ids"]]
        reranked_ids = [int(movie_id) for movie_id in item["reranked_ids"]]
        relevance = {}
        for movie_id in tmdb_ids:
            label = labels_by_pair.get((source_id, movie_id))
            if label is None or label.get("relevance", "").strip() not in {"0", "1", "2"}:
                raise ValueError(
                    f"Missing valid relevance label for {item['source_title']} / {movie_id}"
                )
            relevance[movie_id] = int(label["relevance"])

        tmdb_ndcg = ndcg_at_k(tmdb_ids, relevance)
        reranked_ndcg = ndcg_at_k(reranked_ids, relevance)
        if tmdb_ndcg is None:
            continue
        delta = reranked_ndcg - tmdb_ndcg
        if only_regressions and delta >= 0:
            continue

        movies = {int(movie["id"]): movie for movie in item["candidates"]}
        scores = candidate_scores(item["source_movie"], item["candidates"], recommender)
        tmdb_ranks = {movie_id: rank for rank, movie_id in enumerate(tmdb_ids, start=1)}
        reranked_ranks = {
            movie_id: rank for rank, movie_id in enumerate(reranked_ids, start=1)
        }

        for movie_id in reranked_ids:
            label = labels_by_pair[(source_id, movie_id)]
            movie = movies[movie_id]
            tmdb_rank = tmdb_ranks[movie_id]
            reranker_rank = reranked_ranks[movie_id]
            movie_scores = scores[movie_id]
            output.append(
                {
                    "source": item["source_title"],
                    "candidate": movie.get("title", ""),
                    "llm_relevance": relevance[movie_id],
                    "llm_reason": label.get("reason", ""),
                    "tmdb_rank": tmdb_rank,
                    "reranker_rank": reranker_rank,
                    "rank_movement": tmdb_rank - reranker_rank,
                    "overview_similarity": _round_optional(
                        movie_scores["overview_similarity"]
                    ),
                    "keyword_similarity": _round_optional(
                        movie_scores["keyword_similarity"]
                    ),
                    "genre_similarity": _round_optional(movie_scores["genre_similarity"]),
                    "collection_match": _round_optional(movie_scores["collection_match"]),
                    "final_score": round(movie_scores["final_score"], 4),
                    "issue_type": diagnostic_issue(
                        relevance[movie_id], tmdb_rank, reranker_rank
                    ),
                    "needs_review": label.get("needs_review", ""),
                    "ndcg_difference": round(delta, 4),
                }
            )

    return output


def _round_optional(value: float | None) -> float | str:
    return "" if value is None else round(value, 4)


def diagnose(
    run_dir: Path,
    output_path: Path | None = None,
    snapshot_path: Path | None = None,
    only_regressions: bool = False,
) -> Path:
    from src.embeddings import get_embedding_model, load_config
    from src.recommender import MovieRecommender

    custom_snapshot = snapshot_path is not None
    snapshot_path = snapshot_path or run_dir / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    with (run_dir / "labels.csv").open(newline="", encoding="utf-8-sig") as stream:
        label_rows = list(csv.DictReader(stream))

    config = load_config()
    recommender = MovieRecommender(
        get_embedding_model(config),
        weights=config.get("ranking", {}).get("weights"),
    )
    rows = build_diagnostic_rows(
        snapshot,
        label_rows,
        recommender,
        only_regressions=only_regressions,
    )
    if not rows:
        raise ValueError("No movies matched the requested diagnostic filters")

    default_name = "diagnostics_four_signals.csv" if custom_snapshot else "diagnostics.csv"
    output_path = output_path or run_dir / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    affected_sources = len({row["source"] for row in rows})
    issues = sum(bool(row["issue_type"]) for row in rows)
    print(
        f"Wrote {len(rows)} candidates across {affected_sources} source movies to "
        f"{output_path}; {issues} rows have a diagnostic issue."
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--sources", type=Path, default=Path("evaluation/sources.json"))
    prepare_cmd.add_argument("--out", type=Path, required=True)
    score_cmd = commands.add_parser("score")
    score_cmd.add_argument("--run", type=Path, required=True)
    score_cmd.add_argument("--snapshot", type=Path)
    label_cmd = commands.add_parser("auto-label")
    label_cmd.add_argument("--run", type=Path, required=True)
    label_cmd.add_argument("--provider", choices=("gemini", "local"), default="gemini")
    label_cmd.add_argument("--model", required=True, help="Judge model ID for the selected provider")
    label_cmd.add_argument("--passes", type=int, default=1, choices=(1, 2))
    label_cmd.add_argument("--refresh-llm", action="store_true")
    diagnose_cmd = commands.add_parser("diagnose")
    diagnose_cmd.add_argument("--run", type=Path, required=True)
    diagnose_cmd.add_argument("--out", type=Path)
    diagnose_cmd.add_argument("--snapshot", type=Path)
    diagnose_cmd.add_argument("--only-regressions", action="store_true")
    enrich_cmd = commands.add_parser("enrich")
    enrich_cmd.add_argument("--run", type=Path, required=True)
    enrich_cmd.add_argument("--out", type=Path)
    rerank_cmd = commands.add_parser("rerank")
    rerank_cmd.add_argument("--run", type=Path, required=True)
    rerank_cmd.add_argument("--snapshot", type=Path)
    rerank_cmd.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        asyncio.run(prepare(args.sources, args.out))
    elif args.command == "score":
        score(args.run, snapshot_path=args.snapshot)
    elif args.command == "auto-label":
        auto_label(args.run, args.model, args.passes, args.refresh_llm, args.provider)
    elif args.command == "diagnose":
        diagnose(
            args.run,
            output_path=args.out,
            snapshot_path=args.snapshot,
            only_regressions=args.only_regressions,
        )
    elif args.command == "rerank":
        rerank(args.run, output_path=args.out, snapshot_path=args.snapshot)
    else:
        asyncio.run(enrich_snapshot(args.run, output_path=args.out))


if __name__ == "__main__":
    main()
