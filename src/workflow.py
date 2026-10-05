"""LangGraph orchestration for the movie assistant."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.preferences import extract_preferences
from src.router import Intent, classify_intent, extract_title, extract_year

logger = logging.getLogger(__name__)


class MovieWorkflowState(TypedDict, total=False):
    query: str
    region: str
    intent: str
    title: str | None
    year: int | None
    preferences: dict[str, Any]
    recommendation_mode: str
    result: dict[str, Any]
    explanation_context: dict[str, Any]
    fallback_answer: str
    generated_answer: str
    generation_attempted: bool
    generation_error: str
    validation_passed: bool
    workflow_steps: list[str]


class LangGraphMovieAssistant:
    """Route retrieval, explanation, validation, and fallback explicitly."""

    _TITLE_INTENTS = {
        Intent.WATCH_PROVIDERS,
        Intent.MOVIE_DETAILS,
        Intent.MOVIE_ENDING,
        Intent.SIMILAR_MOVIES,
    }

    def __init__(self, core_assistant):
        self.core_assistant = core_assistant
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(MovieWorkflowState)
        graph.add_node("understand_request", self._understand_request)
        graph.add_node("request_clarification", self._request_clarification)
        graph.add_node("run_trending", self._run_trending)
        graph.add_node("run_watch_providers", self._run_watch_providers)
        graph.add_node("run_movie_details", self._run_movie_details)
        graph.add_node("run_similar_movie", self._run_similar_movie)
        graph.add_node("run_preference_request", self._run_preference_request)
        graph.add_node("run_general_question", self._run_general_question)
        graph.add_node("generate_explanation", self._generate_explanation)
        graph.add_node("validate_explanation", self._validate_explanation)
        graph.add_node("deterministic_fallback", self._deterministic_fallback)

        graph.add_edge(START, "understand_request")
        graph.add_conditional_edges(
            "understand_request",
            self._select_route,
            {
                "clarification": "request_clarification",
                "trending": "run_trending",
                "watch_providers": "run_watch_providers",
                "movie_details": "run_movie_details",
                "similar_movie": "run_similar_movie",
                "preference_request": "run_preference_request",
                "general_question": "run_general_question",
            },
        )
        graph.add_edge("request_clarification", "deterministic_fallback")
        for route_node in (
            "run_trending",
            "run_watch_providers",
            "run_movie_details",
            "run_similar_movie",
            "run_preference_request",
            "run_general_question",
        ):
            graph.add_edge(route_node, "generate_explanation")
        graph.add_edge("generate_explanation", "validate_explanation")
        graph.add_conditional_edges(
            "validate_explanation",
            self._select_output,
            {"valid": END, "fallback": "deterministic_fallback"},
        )
        graph.add_edge("deterministic_fallback", END)
        return graph.compile()

    @staticmethod
    def _understand_request(state: MovieWorkflowState) -> MovieWorkflowState:
        query = state["query"]
        intent = classify_intent(query)
        title = extract_title(query, intent)
        return {
            "intent": intent.value,
            "title": title,
            "year": extract_year(query),
            "preferences": extract_preferences(query),
            "recommendation_mode": (
                "preference_discovery"
                if intent == Intent.RECOMMENDATION
                else "similar_to_movie"
                if intent == Intent.SIMILAR_MOVIES
                else ""
            ),
            "workflow_steps": ["understand_request"],
        }

    def _select_route(self, state: MovieWorkflowState) -> str:
        intent = Intent(state["intent"])
        if intent in self._TITLE_INTENTS and not state.get("title"):
            return "clarification"
        return {
            Intent.TRENDING: "trending",
            Intent.WATCH_PROVIDERS: "watch_providers",
            Intent.MOVIE_DETAILS: "movie_details",
            Intent.MOVIE_ENDING: "movie_details",
            Intent.SIMILAR_MOVIES: "similar_movie",
            Intent.RECOMMENDATION: "preference_request",
            Intent.GENERAL: "general_question",
        }[intent]

    @staticmethod
    def _select_output(state: MovieWorkflowState) -> str:
        return "valid" if state.get("validation_passed") else "fallback"

    @staticmethod
    def _append_step(state: MovieWorkflowState, step: str) -> list[str]:
        return [*state.get("workflow_steps", []), step]

    def _request_clarification(self, state: MovieWorkflowState) -> MovieWorkflowState:
        answer = "Which movie are you referring to?"
        return {
            "result": {
                "answer": answer,
                "route": "clarification",
                "intent": state["intent"],
                "tools_used": [],
                "sources": [],
            },
            "fallback_answer": answer,
            "workflow_steps": self._append_step(state, "request_clarification"),
        }

    async def _run_trending(self, state: MovieWorkflowState) -> MovieWorkflowState:
        return await self._execute(state, Intent.TRENDING, "run_trending")

    async def _run_watch_providers(self, state: MovieWorkflowState) -> MovieWorkflowState:
        return await self._execute(state, Intent.WATCH_PROVIDERS, "run_watch_providers")

    async def _run_movie_details(self, state: MovieWorkflowState) -> MovieWorkflowState:
        return await self._execute(
            state,
            Intent(state["intent"]),
            "run_movie_details",
        )

    async def _run_similar_movie(self, state: MovieWorkflowState) -> MovieWorkflowState:
        return await self._execute(state, Intent.SIMILAR_MOVIES, "run_similar_movie")

    async def _run_general_question(self, state: MovieWorkflowState) -> MovieWorkflowState:
        return await self._execute(state, Intent.GENERAL, "run_general_question")

    async def _run_preference_request(self, state: MovieWorkflowState) -> MovieWorkflowState:
        prepare = getattr(self.core_assistant, "prepare_preference", None)
        if prepare is not None:
            result = await prepare(state["query"], state["preferences"])
        else:
            result = await self.core_assistant.answer_preference(
                state["query"], state["preferences"]
            )
        return self._store_result(state, result, "run_preference_request")

    async def _execute(
        self,
        state: MovieWorkflowState,
        intent: Intent,
        step: str,
    ) -> MovieWorkflowState:
        prepare = getattr(self.core_assistant, "prepare_for_intent", None)
        if prepare is not None:
            result = await prepare(state["query"], intent, state.get("region", "US"))
        else:
            result = await self.core_assistant.answer_for_intent(
                state["query"], intent, state.get("region", "US")
            )
        return self._store_result(state, result, step)

    def _store_result(
        self,
        state: MovieWorkflowState,
        result: dict[str, Any],
        step: str,
    ) -> MovieWorkflowState:
        clean_result = dict(result)
        context = clean_result.pop("_explanation_context", None)
        answer = str(clean_result.get("answer", "")).strip()
        return {
            "result": clean_result,
            "fallback_answer": answer,
            "explanation_context": context or {"kind": "grounded", "verified_answer": answer},
            "workflow_steps": self._append_step(state, step),
        }

    async def _generate_explanation(self, state: MovieWorkflowState) -> MovieWorkflowState:
        generator = getattr(self.core_assistant, "generate_explanation", None)
        steps = self._append_step(state, "generate_explanation")

        if not state.get("explanation_context", {}).get("allow_synthesis", True):
            return {
                "generated_answer": state.get("fallback_answer", ""),
                "generation_attempted": False,
                "workflow_steps": steps,
            }

        # Test doubles and compatibility integrations may already return their
        # final explanation and have no separate generator.
        if generator is None:
            return {
                "generated_answer": state.get("fallback_answer", ""),
                "generation_attempted": False,
                "workflow_steps": steps,
            }

        try:
            answer = await asyncio.to_thread(
                generator, state["query"], state["explanation_context"]
            )
            return {
                "generated_answer": answer,
                "generation_attempted": True,
                "workflow_steps": steps,
            }
        except RuntimeError as exc:
            logger.warning("Fine-tuned Gemma explanation unavailable: %s", exc)
            return {
                "generated_answer": "",
                "generation_attempted": True,
                "generation_error": str(exc),
                "workflow_steps": steps,
            }
        except Exception as exc:
            logger.exception("Fine-tuned Gemma explanation failed")
            return {
                "generated_answer": "",
                "generation_attempted": True,
                "generation_error": str(exc),
                "workflow_steps": steps,
            }

    def _validate_explanation(self, state: MovieWorkflowState) -> MovieWorkflowState:
        answer = str(state.get("generated_answer", "")).strip()
        valid = bool(answer) and not state.get("generation_error")
        validator = getattr(self.core_assistant, "validate_explanation", None)

        if valid and validator is not None and state.get("generation_attempted"):
            try:
                validator(answer, state["explanation_context"])
            except Exception as exc:
                logger.warning("Generated explanation failed validation: %s", exc)
                valid = False

        result = dict(state.get("result") or {})
        steps = self._append_step(state, "validate_explanation")
        if valid:
            result["answer"] = answer
            if state.get("generation_attempted"):
                sources = list(result.get("sources") or [])
                if "Fine-tuned Gemma response synthesis" not in sources:
                    sources.append("Fine-tuned Gemma response synthesis")
                result["sources"] = sources
            result["workflow_steps"] = steps

        return {
            "result": result,
            "validation_passed": valid,
            "workflow_steps": steps,
        }

    def _deterministic_fallback(self, state: MovieWorkflowState) -> MovieWorkflowState:
        result = dict(state.get("result") or {})
        answer = state.get("fallback_answer", "").strip()
        if not answer:
            answer = "I couldn't produce a movie response. Please try again."
            result.setdefault("route", "workflow_error")
            result.setdefault("intent", state.get("intent"))
            result.setdefault("tools_used", [])

        sources = [
            source
            for source in result.get("sources") or []
            if source != "Fine-tuned Gemma response synthesis"
        ]
        if result.get("route") != "clarification" and (
            "Deterministic response formatting" not in sources
        ):
            sources.append("Deterministic response formatting")

        steps = self._append_step(state, "deterministic_fallback")
        result.update({"answer": answer, "sources": sources, "workflow_steps": steps})
        return {"result": result, "workflow_steps": steps}

    async def answer(self, query: str, region: str = "US") -> dict[str, Any]:
        final_state = await self.graph.ainvoke(
            {"query": query, "region": region.upper(), "workflow_steps": []}
        )
        return final_state["result"]
