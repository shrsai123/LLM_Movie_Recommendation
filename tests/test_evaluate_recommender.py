import csv
import json

from scripts import evaluate_recommender as evaluation


def test_judge_prompt_contains_only_movie_content():
    source = {"id": 1, "title": "Source", "overview": "An overview", "genre_ids": [878], "score": 0.9}
    candidate = {"id": 2, "title": "Candidate", "overview": "Another overview", "genre_ids": [18], "rank": 1}

    prompt = evaluation.judge_prompt(source, candidate)

    assert "Science Fiction" in prompt
    assert "Drama" in prompt
    assert '"score"' not in prompt
    assert '"rank"' not in prompt


def test_parse_judgment_requires_valid_rubric_value():
    assert evaluation.parse_judgment('Result: {"relevance": 2, "reason": "Shared isolation theme"}') == {
        "relevance": 2,
        "reason": "Shared isolation theme",
    }


def test_run_judge_retries_after_prose_response():
    responses = iter([
        [{"generated_text": "I will compare the movies first."}],
        [{"generated_text": '{"relevance": 1, "reason": "Shared space setting"}'}],
    ])

    result = evaluation.run_judge(
        lambda *_args, **_kwargs: next(responses),
        {"title": "Source", "overview": "", "genre_ids": []},
        {"title": "Candidate", "overview": "", "genre_ids": []},
    )

    assert result == {"relevance": 1, "reason": "Shared space setting"}


def test_gemini_judge_uses_structured_output():
    calls = []

    class Client:
        def post(self, *args, **kwargs):
            calls.append((args, kwargs))
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "is_error": False,
                    "json": lambda self: {
                        "candidates": [
                            {"content": {"parts": [{"text": '{"relevance": 2, "reason": "Shared space survival themes"}'}]}}
                        ]
                    },
                },
            )()

    judge = evaluation.GeminiJudge(
        client=Client(),
        api_key="test-key",
        model_id="gemini-3.8-flash",
    )
    result = evaluation.run_judge(
        judge,
        {"title": "Source", "overview": "", "genre_ids": [878], "score": 0.9},
        {"title": "Candidate", "overview": "", "genre_ids": [18], "rank": 1},
    )

    assert result["relevance"] == 2
    request_url, request = calls[0]
    assert request_url[0].endswith(":generateContent")
    assert request["json"]["generationConfig"]["responseFormat"]["text"]["mimeType"] == "application/json"
    assert request["json"]["generationConfig"]["responseFormat"]["text"]["schema"] == evaluation.JUDGMENT_SCHEMA
    assert '"score"' not in request["json"]["contents"][0]["parts"][0]["text"]
    assert '"rank"' not in request["json"]["contents"][0]["parts"][0]["text"]


def test_auto_label_preserves_existing_manual_rating(tmp_path, monkeypatch):
    snapshot = [{
        "source_id": "1",
        "source_movie": {"id": 1, "title": "Source", "overview": "Source plot", "genre_ids": [878]},
        "candidates": [{"id": 2, "title": "Candidate", "overview": "Candidate plot", "genre_ids": [18]}],
    }]
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    with (tmp_path / "labels.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "candidate_id", "relevance"])
        writer.writeheader()
        writer.writerow({"source_id": "1", "candidate_id": "2", "relevance": "2"})

    monkeypatch.setattr(evaluation, "load_judge", lambda _model, _provider: object())
    monkeypatch.setattr(evaluation, "run_judge", lambda *_args: (_ for _ in ()).throw(AssertionError()))

    evaluation.auto_label(tmp_path, "judge-model", passes=1, refresh_llm=False)

    with (tmp_path / "labels.csv").open(newline="", encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["relevance"] == "2"
    assert row["label_source"] == ""


def test_rerank_snapshot_refreshes_ids_without_changing_candidates():
    snapshot = [
        {
            "source_id": "1",
            "source_movie": {"id": 1, "title": "Source"},
            "candidates": [
                {"id": 2, "title": "First"},
                {"id": 3, "title": "Second"},
            ],
            "tmdb_ids": [2, 3],
            "reranked_ids": [2, 3],
        }
    ]

    class ReverseRecommender:
        def rank(self, _source, candidates, limit):
            return list(reversed(candidates))[:limit]

    result = evaluation.rerank_snapshot(snapshot, ReverseRecommender())

    assert result[0]["reranked_ids"] == [3, 2]
    assert result[0]["previous_reranked_ids"] == [2, 3]
    assert result[0]["candidates"] == snapshot[0]["candidates"]
