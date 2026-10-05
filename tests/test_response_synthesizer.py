import json

import pytest

from src.response_synthesizer import (
    ResponseSynthesizer,
    SynthesisError,
    build_grounded_messages,
    build_messages,
)

RECOMMENDATIONS = [
    {"title": "Ad Astra", "release_year": 2019},
    {"title": "Moon", "release_year": 2009},
]


def test_validation_accepts_the_locked_ranked_list():
    ResponseSynthesizer.validate(
        "1. Ad Astra (2019)\nA reflective space journey.\n\n"
        "2. Moon (2009)\nA similarly isolated science-fiction story.",
        RECOMMENDATIONS,
    )


@pytest.mark.parametrize(
    "answer",
    [
        "1. Moon (2009)\nExplanation.\n\n2. Ad Astra (2019)\nExplanation.",
        "1. Ad Astra (2019)\nExplanation.",
        "1. Ad Astra (2019)\nExplanation.\n\n2. Solaris (1972)\nExplanation.",
        "1. Ad Astra (2019)\nExplanation.\n\n3. Moon (2009)\nExplanation.",
    ],
)
def test_validation_rejects_selection_or_order_changes(answer):
    with pytest.raises(SynthesisError):
        ResponseSynthesizer.validate(answer, RECOMMENDATIONS)


def test_prompt_keeps_ranked_movies_immutable():
    messages = build_messages("Movies like Moon", RECOMMENDATIONS)
    packet = json.loads(messages[1]["content"].split("\n", 1)[1])

    assert "final and immutable" in messages[0]["content"]
    assert [movie["rank"] for movie in packet["recommendations"]] == [1, 2]
    assert [movie["title"] for movie in packet["recommendations"]] == [
        "Ad Astra",
        "Moon",
    ]
    assert "explicitly name the source movie" in messages[0]["content"]


def test_normalize_keeps_generated_explanations_and_removes_trailing_text():
    raw_answer = (
        "These are strong options:\n\n"
        "1. Ad Astra (2019)\nA reflective space journey.\n\n"
        "2. Moon (2009)\nA similarly isolated science-fiction story.\n\n"
        "Unrelated trailing model output.\n3. An invented movie"
    )

    normalized = ResponseSynthesizer.normalize(raw_answer, RECOMMENDATIONS)

    assert normalized == (
        "1. Ad Astra (2019)\nA reflective space journey.\n\n"
        "2. Moon (2009)\nA similarly isolated science-fiction story."
    )


def test_normalize_rejects_reordered_movies():
    raw_answer = (
        "2. Moon (2009)\nExplanation.\n\n"
        "1. Ad Astra (2019)\nExplanation."
    )

    with pytest.raises(SynthesisError):
        ResponseSynthesizer.normalize(raw_answer, RECOMMENDATIONS)


def test_source_comparison_validator_requires_source_in_each_explanation():
    valid = (
        "1. Ad Astra (2019)\nLike Moon, it explores isolation.\n\n"
        "2. Moon (2009)\nIt resembles Moon through its isolated setting."
    )
    ResponseSynthesizer.validate_source_comparisons(
        valid,
        RECOMMENDATIONS,
        "Moon",
    )

    with pytest.raises(SynthesisError, match="did not compare"):
        ResponseSynthesizer.validate_source_comparisons(
            "1. Ad Astra (2019)\nA space journey.\n\n"
            "2. Moon (2009)\nLike Moon, it explores isolation.",
            RECOMMENDATIONS,
            "Moon",
        )


def test_grounded_prompt_and_validator_preserve_verified_tmdb_answer():
    verified = "Streaming: Max\nTMDB: https://www.themoviedb.org/movie/123"
    messages = build_grounded_messages("Where can I watch it?", verified)
    packet = json.loads(messages[1]["content"].split("\n", 1)[1])

    assert packet["verified_response"] == verified
    ResponseSynthesizer.validate_grounded(
        verified + "\n\nThese are the currently verified options.",
        verified,
    )

    with pytest.raises(SynthesisError):
        ResponseSynthesizer.validate_grounded("Streaming: Netflix", verified)


@pytest.mark.parametrize(
    "suffix",
    [
        "lessly " * 12,
        "lessly" * 12,
    ],
)
def test_grounded_validator_rejects_repetition_loops(suffix):
    verified = "I couldn't find any TMDB movies matching that query."

    with pytest.raises(SynthesisError, match="repetition loop"):
        ResponseSynthesizer.validate_grounded(f"{verified} {suffix}", verified)
