from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    secret_word: str = Field(min_length=1)
    imposter_hint: str = Field(min_length=1)
    human_clue: str = Field(min_length=1)


class BenchmarkConfig(BaseModel):
    backend_url: str
    mode: str
    prompt_technique: str
    repetitions: int = Field(ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)


class RoundLatencyRecord(BaseModel):
    mode: str
    prompt_technique: str
    round_id: str | None
    secret_word: str
    imposter_hint: str
    status: str
    success: bool
    total_latency_ms: float
    create_latency_ms: float | None = None
    continuation_latency_ms: float | None = None
    poll_count: int = 0
    generated_agent_clue_count: int = 0
    playing_order: list[str] = Field(default_factory=list)
    error: str | None = None


class BenchmarkSummary(BaseModel):
    mode: str
    prompt_technique: str
    repetitions: int
    case_count: int
    round_count: int
    success_rate: float
    generation_failed_rate: float
    timeout_rate: float
    average_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    average_create_latency_ms: float
    average_continuation_latency_ms: float
    average_generated_agent_clue_count: float
