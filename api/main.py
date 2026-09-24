import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

def build_assistant():
    from src.bootstrap import build_assistant as assemble_assistant
    return assemble_assistant()

def _cors_origins() -> list[str]:
    configured = os.environ.get("CORS_ORIGINS", "http://localhost:5173")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the application...")
    app.state.assistant = await asyncio.to_thread(build_assistant)
    logger.info("Shutting down the application...")
    yield

app = FastAPI(lifespan=lifespan, title="Movie Blasters API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    started_at = time.perf_counter()

    try:
        result = await request.app.state.assistant.answer(
            payload.message,
            region=payload.region.upper(),
        )
    except Exception as exc:
        logger.exception("Movie assistant request failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the movie request",
        ) from exc

    return ChatResponse(
        answer=result["answer"],
        route=result["route"],
        intent=result.get("intent"),
        tools_used=result.get("tools_used", []),
        sources=result.get("sources", []),
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )