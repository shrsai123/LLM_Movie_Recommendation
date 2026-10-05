"""Generate grounded explanations from the recommender's final ranked movies."""

from __future__ import annotations

import json
import re
from pathlib import Path

# The model is deliberately told not to select or rerank movies. Ranking belongs
# to MovieRecommender and DiversityReranker; this model only verbalizes results.
SYSTEM_PROMPT = """You are the response writer for a movie recommender.
Use only the supplied recommendation data.
The supplied recommendations and their rank order are final and immutable.
Do not select, remove, add, replace, or reorder movies.
Do not introduce any movie, person, score, or fact that is not in the data.
Explain why every recommended movie fits the user's request.
For source-based recommendations, explicitly name the source movie in every
paragraph and describe the concrete plot, theme, keyword, or genre overlap.
Do not print internal scores or a `Signals:` list, and do not merely restate the
recommended movie's overview.
Use this exact heading format for every movie: `1. Title (Year)`.
Write one short, useful paragraph below each heading."""

GROUNDED_SYSTEM_PROMPT = """You are the response writer for a movie assistant.
Use only the supplied verified response; it is the authoritative source of facts.
Preserve every name, title, year, number, provider, and URL exactly.
Start by reproducing the verified response verbatim.
You may then add one short explanation, but do not introduce any new facts."""


class SynthesisError(RuntimeError):
    """Raised when model loading, generation, or output validation fails."""


def _movie_payload(movie: dict, rank: int | None = None) -> dict:
    """Keep only fields that the model is allowed to mention."""
    payload = {
        "title": movie.get("title"),
        "year": movie.get("release_year"),
        "overview": movie.get("overview"),
        "genres": movie.get("genres") or [],
        "keywords": movie.get("keywords") or [],
        "reason": movie.get("reason"),
        "comparison": movie.get("comparison"),
        "score": movie.get("score"),
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def build_messages(
    query: str,
    recommendations: list[dict],
    source_movie: dict | None = None,
    preferences: dict | None = None,
) -> list[dict[str, str]]:
    """Build the same prompt structure for training and production inference."""
    packet = {
        "user_request": query,
        "source_movie": _movie_payload(source_movie) if source_movie else None,
        "preferences": preferences or None,
        "recommendations": [
            _movie_payload(movie, rank=index)
            for index, movie in enumerate(recommendations, start=1)
        ],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Write the final recommendation response from this data:\n"
            + json.dumps(packet, ensure_ascii=False, indent=2),
        },
    ]


def build_grounded_messages(query: str, verified_answer: str) -> list[dict[str, str]]:
    """Build a grounded prompt for TMDB-backed non-recommendation routes."""
    packet = {
        "user_request": query,
        "verified_response": verified_answer,
    }
    return [
        {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Explain this verified movie response:\n"
            + json.dumps(packet, ensure_ascii=False, indent=2),
        },
    ]


class ResponseSynthesizer:
    """Load base Gemma and optionally attach a trained LoRA adapter."""

    def __init__(
        self,
        base_model: str = "google/gemma-3-1b-it",
        adapter_path: str | None = None,
        quantize_4bit: bool = True,
        max_new_tokens: int = 250,
        device: str = "auto",
    ):
        # Heavy ML libraries are imported lazily so normal unit tests do not need
        # to load Transformers, PEFT, or model weights.
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise SynthesisError(
                "Install the packages in requirements-training.txt"
            ) from exc

        tokenizer_source = adapter_path or base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token

        if device not in {"auto", "cpu", "cuda"}:
            raise SynthesisError("Synthesis device must be auto, cpu, or cuda")

        use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        model_kwargs = {
            "device_map": {"": 0} if use_cuda else {"": "cpu"},
        }
        if quantize_4bit and use_cuda:
            compute_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            model_kwargs["dtype"] = (
                torch.bfloat16 if use_cuda else torch.float32
            )

        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if adapter_path:
            adapter = Path(adapter_path)
            if not adapter.exists():
                raise SynthesisError(f"Synthesis adapter not found: {adapter}")
            model = PeftModel.from_pretrained(model, adapter)

        self.model = model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        query: str,
        recommendations: list[dict],
        source_movie: dict | None = None,
        preferences: dict | None = None,
        validate_output: bool = True,
    ) -> str:
        """Generate and validate one grounded recommendation response."""
        if not recommendations:
            raise SynthesisError("At least one recommendation is required")

        messages = build_messages(
            query=query,
            source_movie=source_movie,
            preferences=preferences,
            recommendations=recommendations,
        )
        answer = self._generate_messages(messages)
        if validate_output:
            answer = self.normalize(answer, recommendations)
            self.validate(answer, recommendations)
            if source_movie:
                self.validate_source_comparisons(
                    answer,
                    recommendations,
                    str(source_movie.get("title") or ""),
                )
        return answer

    def generate_grounded(
        self,
        query: str,
        verified_answer: str,
        validate_output: bool = True,
    ) -> str:
        """Explain a TMDB-backed answer without changing its verified facts."""
        if not verified_answer.strip():
            raise SynthesisError("A verified response is required")

        answer = self._generate_messages(build_grounded_messages(query, verified_answer))
        if validate_output:
            self.validate_grounded(answer, verified_answer)
        return answer

    def _generate_messages(self, messages: list[dict[str, str]]) -> str:
        """Run deterministic generation for an already constructed chat prompt."""
        import torch

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)

        # Greedy generation makes evaluation reproducible and prevents sampling
        # differences from obscuring the base-vs-adapter comparison.
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def validate_grounded(answer: str, verified_answer: str) -> None:
        """Require the model to preserve the complete authoritative response."""
        if not answer:
            raise SynthesisError("The synthesis model returned an empty response")
        if verified_answer.strip() not in answer:
            raise SynthesisError("Generated response changed or omitted verified facts")
        if ResponseSynthesizer._has_repetition_loop(answer):
            raise SynthesisError("Generated response contains a repetition loop")

    @staticmethod
    def _has_repetition_loop(answer: str) -> bool:
        """Detect the common spaced and concatenated decoder repetition loops."""
        if re.search(
            r"\b([\w'-]{3,})\b(?:\s+\1\b){4,}",
            answer,
            flags=re.IGNORECASE,
        ):
            return True
        return bool(re.search(r"(.{4,24})\1{4,}", answer, flags=re.IGNORECASE))

    @staticmethod
    def normalize(answer: str, recommendations: list[dict]) -> str:
        """Keep one generated explanation for each immutable ranked movie."""
        if not answer:
            raise SynthesisError("The synthesis model returned an empty response")

        matches = []
        for rank, movie in enumerate(recommendations, start=1):
            title = str(movie.get("title") or "").strip()
            if not title:
                raise SynthesisError("A recommendation is missing its title")

            year = movie.get("release_year")
            label = f"{title} ({year})" if year else title
            match = re.search(
                rf"(?mi)^\s*{rank}\.\s+{re.escape(label)}\s*$",
                answer,
            )
            if match is None:
                raise SynthesisError(f"Generated response omitted: {label}")
            matches.append((label, match))

        positions = [match.start() for _, match in matches]
        if positions != sorted(positions):
            raise SynthesisError("Generated response changed the recommendation order")

        sections = []
        for index, (label, match) in enumerate(matches):
            end = matches[index + 1][1].start() if index + 1 < len(matches) else len(answer)
            body = answer[match.end() : end]
            explanation_lines = []
            for raw_line in body.splitlines():
                line = raw_line.strip()
                if not line:
                    if explanation_lines:
                        break
                    continue
                if re.match(r"^\d+\.\s+", line):
                    break
                explanation_lines.append(line)

            explanation = " ".join(explanation_lines)
            if not explanation:
                raise SynthesisError(f"Generated response did not explain: {label}")
            sections.append(f"{index + 1}. {label}\n{explanation}")

        return "\n\n".join(sections)

    @staticmethod
    def validate(answer: str, recommendations: list[dict]) -> None:
        """Require numbered headings for exactly the supplied ranked movies."""
        if not answer:
            raise SynthesisError("The synthesis model returned an empty response")

        headings = re.findall(r"(?m)^\s*(\d+)\.\s+(.+?)\s*$", answer)
        if len(headings) != len(recommendations):
            raise SynthesisError(
                "Generated response changed the recommendation selection"
            )

        for expected_rank, (movie, heading) in enumerate(
            zip(recommendations, headings, strict=True),
            start=1,
        ):
            title = str(movie.get("title") or "").strip()
            if not title:
                raise SynthesisError("A recommendation is missing its title")

            rank_text, generated_label = heading
            if int(rank_text) != expected_rank:
                raise SynthesisError("Generated response changed the recommendation order")

            year = movie.get("release_year")
            expected_label = f"{title} ({year})" if year else title
            if generated_label.strip().casefold() != expected_label.casefold():
                raise SynthesisError(
                    "Generated response changed the recommendation selection or order"
                )

    @staticmethod
    def validate_source_comparisons(
        answer: str,
        recommendations: list[dict],
        source_title: str,
    ) -> None:
        """Require every source-based recommendation to make the comparison explicit."""
        if not source_title.strip():
            raise SynthesisError("A source movie title is required for comparison")
        if "signals:" in answer.casefold():
            raise SynthesisError("Generated response exposed internal ranking signals")

        headings = list(re.finditer(r"(?m)^\s*\d+\.\s+.+?\s*$", answer))
        if len(headings) != len(recommendations):
            raise SynthesisError("Generated response changed the recommendation selection")
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(answer)
            explanation = answer[heading.end() : end]
            if source_title.casefold() not in explanation.casefold():
                raise SynthesisError(
                    f"Generated response did not compare recommendation to {source_title}"
                )
