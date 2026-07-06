from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Iterator
from uuid import uuid4


@dataclass
class StageMetric:
    name: str
    duration_ms: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RequestMetric:
    id: str
    method: str
    path: str
    status_code: int
    started_at: datetime
    duration_ms: float
    stages: list[StageMetric] = field(default_factory=list)


@dataclass
class ActiveRequestTrace:
    id: str
    method: str
    path: str
    started_at: datetime
    start_time: float
    stages: list[StageMetric] = field(default_factory=list)


MAX_REQUEST_METRICS = 200
_current_trace: ContextVar[ActiveRequestTrace | None] = ContextVar(
    "current_request_trace",
    default=None,
)
_recent_requests: deque[RequestMetric] = deque(maxlen=MAX_REQUEST_METRICS)


def start_request_trace(method: str, path: str) -> tuple[ActiveRequestTrace, object]:
    trace = ActiveRequestTrace(
        id=str(uuid4()),
        method=method,
        path=path,
        started_at=datetime.now(UTC),
        start_time=perf_counter(),
    )
    token = _current_trace.set(trace)
    return trace, token


def finish_request_trace(
    trace: ActiveRequestTrace,
    token: object,
    status_code: int,
) -> None:
    duration_ms = (perf_counter() - trace.start_time) * 1000
    _recent_requests.appendleft(
        RequestMetric(
            id=trace.id,
            method=trace.method,
            path=trace.path,
            status_code=status_code,
            started_at=trace.started_at,
            duration_ms=duration_ms,
            stages=list(trace.stages),
        )
    )
    _current_trace.reset(token)


@contextmanager
def measure_stage(name: str, metadata: dict[str, str] | None = None) -> Iterator[None]:
    start_time = perf_counter()
    try:
        yield
    finally:
        trace = _current_trace.get()
        if trace is not None:
            trace.stages.append(
                StageMetric(
                    name=name,
                    duration_ms=(perf_counter() - start_time) * 1000,
                    metadata=metadata or {},
                )
            )


def list_recent_request_metrics() -> list[RequestMetric]:
    return list(_recent_requests)


def clear_request_metrics() -> None:
    _recent_requests.clear()
