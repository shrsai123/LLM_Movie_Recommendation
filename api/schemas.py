from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    region: str = Field(default="US", min_length=2, max_length=2)


class ChatResponse(BaseModel):
    answer: str
    route: str
    intent: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
