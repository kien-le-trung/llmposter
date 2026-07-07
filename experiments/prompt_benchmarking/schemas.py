from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    secret_word: str = Field(min_length=1)
    imposter_hint: str = Field(min_length=1)
    human_clue: str = Field(min_length=1)


class BenchmarkConfig(BaseModel):
    backend_url: str
    technique: str
    repetitions: int = Field(ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)


class ClueBenchmarkRecord(BaseModel):
    technique: str
    round_id: str
    secret_word: str
    imposter_hint: str
    agent_id: str
    agent_name: str
    clue: str
    inference_mode: str


class RoundBenchmarkRecord(BaseModel):
    technique: str
    round_id: str | None
    secret_word: str
    imposter_hint: str
    status: str
    latency_ms: float
    success: bool
    error: str | None = None


class BenchmarkSummary(BaseModel):
    technique: str
    average_latency_ms: float
    round_success_rate: float
    generation_failed_rate: float
    duplicate_clue_rate: float
    secret_word_leak_rate: float
    empty_clue_rate: float
    average_clue_word_count: float
