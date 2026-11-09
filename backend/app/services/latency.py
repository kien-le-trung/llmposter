"""Opt-in, content-free timing events for background work and HTTP inference."""

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
import json
from pathlib import Path
from threading import Lock
from time import perf_counter_ns
from typing import TextIO
from uuid import uuid4

_context: ContextVar[dict] = ContextVar("latency_context", default={})
_stream: TextIO | None = None
_run_id = ""
_lock = Lock()
# Never serialize arbitrary model/provider objects or exception messages.
_FIELDS = set("""
round_id phase agent_id turn_id request_id structured_attempt attempt message_id sequence
model response_model provider routed_provider prompt_characters prompt_tokens completion_tokens total_tokens
status_code outcome reason duration_ms start_ns end_ns started_at delay_ms
capacity client_limit processed_messages publication_sequence snapshot_sequence
playing_order imposter_player_id discussion_seconds error_type warmup
""".split())


def configure(path: str | Path | None, run_id: str = "") -> None:
    global _stream, _run_id
    with _lock:
        if _stream is not None:
            _stream.close()
        _stream = None
        _run_id = run_id
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            _stream = target.open("a", encoding="utf-8", buffering=1)


def emit(event: str, **fields) -> None:
    if _stream is None:
        return
    values = {**_context.get(), **fields}
    row = {"event": event, "event_id": str(uuid4()), "run_id": _run_id,
           "at": datetime.now(UTC).isoformat(), "mono_ns": perf_counter_ns(),
           **{key: value for key, value in values.items() if key in _FIELDS}}
    # A telemetry disk failure must not change gameplay behavior.
    try:
        with _lock:
            if _stream is not None:
                _stream.write(json.dumps(row) + "\n")
    except OSError:
        pass


@contextmanager
def scope(**fields):
    token = _context.set({**_context.get(), **{k: v for k, v in fields.items() if k in _FIELDS}})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def span(event: str, **fields):
    start = perf_counter_ns()
    started_at = datetime.now(UTC).isoformat()
    details = {"outcome": "ok"}
    with scope(**fields):
        try:
            yield details
        except BaseException as exc:
            if details["outcome"] == "ok":
                details["outcome"] = "cancelled" if type(exc).__name__ == "CancelledError" else "failure"
            details["error_type"] = type(exc).__name__
            raise
        finally:
            end = perf_counter_ns()
            emit(event, start_ns=start, end_ns=end, started_at=started_at,
                 duration_ms=(end - start) / 1e6, **details)


@asynccontextmanager
async def acquire(semaphore, event: str, **fields):
    with span(event, **fields):
        await semaphore.acquire()
    try:
        yield
    finally:
        semaphore.release()
