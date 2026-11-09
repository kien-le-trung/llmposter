"""Run with: python experiments/src/runner/run_latency.py --help."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import socket
import platform
import importlib.metadata
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parents[1]
for directory in (SRC, ROOT / "backend"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import httpx  # noqa: E402

from backend_handler.api_client import (  # noqa: E402
    BackendExperimentApiClient, BackendExperimentRequest, BackendExperimentResult,
)
from latency_benchmark.report import read_events, summarize, write_json  # noqa: E402
from runner.backend_process import BackendProcess  # noqa: E402
from runner.config import BackendConfig  # noqa: E402
from tracker.mlflow_tracker import MlflowTracker, TrackerConfig  # noqa: E402


@dataclass(frozen=True)
class Cell:
    kind: str
    capacity: int
    parallel: int
    client_limit: int
    delay: float


def port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Process(BackendProcess):
    """Reuse backend readiness/stop lifecycle with a benchmark-specific app factory."""

    def __init__(self, factory: str, env: dict, log: Path):
        super().__init__(BackendConfig("127.0.0.1", port(), False, 45), None)
        self.factory, self.env, self.log = factory, env, log
        self.log_stream = None

    def start(self):
        self.log_stream = self.log.open("w", encoding="utf-8")
        env = {**os.environ, **self.env}
        env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(ROOT / "backend")])
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", self.factory, "--factory", "--host", "127.0.0.1",
             "--port", str(self.backend_config.port), "--no-access-log", "--log-level", "warning"],
            cwd=ROOT, env=env, stdout=self.log_stream, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def stop(self):
        super().stop()
        if self.log_stream is not None:
            self.log_stream.close()


class LatencyClient(BackendExperimentApiClient):
    """Normal HTTP gameplay, with separate phase deadlines and content-free artifacts."""

    async def play(self, client, run_id, warmup, duration, rounds, observations):
        began = perf_counter()
        row = {"run_id": run_id, "round_id": None, "warmup": warmup,
               "outcome": "failure", "started_at": datetime.now(UTC).isoformat()}
        rounds.append(row)  # Retain partial progress even if the phase is cancelled.
        phase = "clues"
        seen = set()

        async def request(method, path, payload=None, timeout=20):
            async with asyncio.timeout(timeout):
                artifact, body = await self._request_json(client, method, path, json=payload)
            if not artifact.success:
                # Do not export arbitrary API/provider error bodies.
                raise RuntimeError(f"HTTP_{artifact.status_code or 'transport_error'}")
            return body

        try:
            body = await request("POST", "/rounds", {
                "include_human": False, "secret_word": "satellite", "imposter_hint": "orbit",
            })
            row["round_id"] = body["id"]
            row["playing_order"] = ",".join(p["id"] for p in body["playing_order"])
            row["created_at"] = body["created_at"]
            deadline = perf_counter() + 180
            while True:
                now = datetime.now(UTC).isoformat()
                for message in body.get("discussion", {}).get("messages", []):
                    if message["id"] not in seen:
                        seen.add(message["id"])
                        observations.append({"run_id": run_id, "round_id": row["round_id"],
                                             "message_id": message["id"], "warmup": warmup,
                                             "sequence": message["sequence"], "observed_at": now,
                                             "server_created_at": message["created_at"]})
                status = body["status"]
                if status == "generation_failed":
                    raise RuntimeError("generation_failed")
                if status == "awaiting_human_clue":
                    raise RuntimeError("unexpected_human_round")
                if status == "discussion" and phase == "clues":
                    phase = "discussion"
                    deadline = perf_counter() + duration + 30
                if status == "ready_to_vote":
                    break
                if perf_counter() >= deadline:
                    raise TimeoutError(f"{phase}_timeout")
                await asyncio.sleep(self.poll_interval_seconds)
                body = await request("GET", f'/rounds/{row["round_id"]}')
            phase = "voting"
            vote_started = perf_counter()
            await request("POST", f'/rounds/{row["round_id"]}/vote', {}, timeout=180)
            row["voting_ms"] = (perf_counter() - vote_started) * 1000
            row["outcome"] = "complete"
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["failed_phase"] = phase
            # Only internally generated codes, never response bodies or request content.
        finally:
            row["latency_ms"] = (perf_counter() - began) * 1000
        return row


def cells(args):
    delays = args.delays if args.mode == "controlled" else [0]
    result = [Cell("discussion", capacity, parallel, 4, delay)
              for delay in delays for capacity in args.capacities for parallel in args.parallel_rounds]
    if args.mode == "controlled" and not args.skip_client_sweep:
        result += [Cell("client", 4, parallel, limit, delay)
                   for delay in delays for limit in args.client_limits for parallel in args.parallel_rounds]
    random.Random(args.seed).shuffle(result)
    return result


def positive_list(raw):
    values = [int(value) for value in raw.split(",")]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("Use positive comma-separated integers")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["controlled", "live"], default="controlled")
    parser.add_argument("--capacities", type=positive_list)
    parser.add_argument("--parallel-rounds", type=positive_list, default=[1, 4])
    parser.add_argument("--client-limits", type=positive_list, default=[1, 4])
    parser.add_argument("--skip-client-sweep", action="store_true")
    parser.add_argument("--delays", type=lambda raw: [float(x) for x in raw.split(",")], default=[.2, 1.0])
    parser.add_argument("--scenario", choices=["normal", "rate_limit", "invalid", "silence", "timeout", "deadline"], default="normal")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--discussion-seconds", type=int)
    parser.add_argument("--llm-config", type=Path, help="JSON containing llm_config, or its raw object")
    parser.add_argument("--model", help="Live OpenRouter model override")
    parser.add_argument("--database-url-env", help="Environment variable containing a dedicated test DB URL")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/latency/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true", help="Keep local artifacts without MLflow")
    args = parser.parse_args(argv)
    args.capacities = args.capacities or ([1, 4, 8] if args.mode == "live" else [1, 2, 4, 8])
    args.repetitions = args.repetitions if args.repetitions is not None else (1 if args.mode == "live" else 5)
    args.discussion_seconds = args.discussion_seconds if args.discussion_seconds is not None else (90 if args.mode == "live" else 15)
    if args.repetitions < 1 or not 1 <= args.discussion_seconds <= 600 or any(d < 0 for d in args.delays):
        parser.error("Repetitions must be positive, discussion 1–600 seconds, delays nonnegative")
    if args.database_url_env and not os.getenv(args.database_url_env):
        parser.error("The selected database environment variable is unset")
    return args


async def run(args):
    from app.core.config import LLMConfig, Settings
    from app.services.agents.runtime_agents import list_static_agent_configs

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    output = args.output.resolve() / stamp
    output.mkdir(parents=True)
    settings = Settings()
    llm = settings.llm_config
    if args.llm_config:
        raw = json.loads(args.llm_config.read_text(encoding="utf-8"))
        llm = LLMConfig.model_validate(raw.get("llm_config", raw))
    if args.model:
        llm = llm.model_copy(update={"model": args.model})
    if args.mode == "live":
        if llm.provider != "openrouter" or llm.model == "openrouter/free" or llm.model.endswith(":free"):
            raise ValueError("Live mode requires a paid OpenRouter model")
        if not llm.api_key_env or not settings.get_env_value(llm.api_key_env):
            raise ValueError("Live OpenRouter credentials are not configured")
    if args.mode == "live":
        async with httpx.AsyncClient(timeout=20) as check:
            response = await check.get("https://openrouter.ai/api/v1/key", headers={
                "Authorization": "Bearer " + settings.get_env_value(llm.api_key_env)})
            response.raise_for_status()
            key = response.json()["data"]
            remaining = key.get("limit_remaining")
            if key.get("limit_reset") or not isinstance(remaining, (int, float)) or not 0 < remaining <= 5:
                raise RuntimeError("Live pilot requires a dedicated non-resetting key with at most $5 remaining")
    matrix = cells(args)
    params = {"mode": args.mode, "repetitions": args.repetitions,
              "discussion_seconds": args.discussion_seconds, "seed": args.seed,
              "scenario": args.scenario, "matrix": [asdict(c) for c in matrix],
              "database": "external_test_database" if args.database_url_env else "isolated_sqlite",
              "model": llm.model if args.mode == "live" else "latency-stub",
              "agents": [{key: getattr(agent, key) for key in
                          ("id", "temperature", "top_p", "max_tokens", "version")}
                         for agent in list_static_agent_configs()],
              "scheduler": {key: getattr(settings, key) for key in
                            ("discussion_debounce_seconds", "discussion_cooldown_seconds",
                             "discussion_silence_seconds")},
              "token_limits": {"min_max_tokens": llm.min_max_tokens,
                               "structured_max_tokens": llm.structured_max_tokens},
              "prompt_technique": settings.clue_prompt_technique,
              "agent_configuration": "static", "poll_interval_seconds": .25}
    params["python"] = platform.python_version()
    params["dependencies"] = {name: importlib.metadata.version(name) for name in ("httpx", "fastapi", "pydantic", "langgraph")}
    params["revision"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    params["dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    write_json(output / "params.json", params)
    rounds, events, observations = [], [], []
    print(f"Artifacts: {output}", flush=True)
    exit_error = None
    try:
        for index, cell in enumerate(matrix):
            run_id = f"{index:02d}-{cell.kind}-cap{cell.capacity}-rounds{cell.parallel}-client{cell.client_limit}-delay{cell.delay}"
            directory = output / run_id
            directory.mkdir()
            stub = None
            backend = None
            cell_config = None
            print(f"Starting {run_id}", flush=True)
            try:
                model_config = llm.model_copy(update={"max_concurrent_requests": cell.client_limit})
                if args.mode == "controlled":
                    stub = Process("latency_benchmark.stub:create_app", {
                        "LATENCY_STUB_DELAY": str(cell.delay), "LATENCY_STUB_SCENARIO": args.scenario,
                    }, directory / "stub.log")
                    stub.start()
                    await stub.wait_until_healthy()
                    model_config = LLMConfig(
                        name="Local latency stub", provider="local", model="latency-stub",
                        chat_url=stub.backend_url + "/v1/chat/completions",
                        max_concurrent_requests=cell.client_limit,
                    )
                cell_config = {"run_id": run_id, "events_path": str(directory / "events.jsonl"),
                               "settings": {"llm_config": model_config.model_dump(),
                                            "inference_mode": "remote", "agent_config_source": "static",
                                            "clue_prompt_technique": settings.clue_prompt_technique,
                                            "discussion_duration_seconds": args.discussion_seconds,
                                            "discussion_max_concurrent_turns": cell.capacity}}
                config_path = directory / "backend.json"
                write_json(config_path, cell_config)
                db_url = os.getenv(args.database_url_env) if args.database_url_env else (
                    "sqlite:///" + (directory / "game.sqlite").as_posix()
                )
                backend = Process("latency_benchmark.backend:create_app", {
                    "LATENCY_CONFIG": str(config_path), "DATABASE_URL": db_url,
                }, directory / "backend.log")
                backend.start()
                await backend.wait_until_healthy()
                api = LatencyClient(backend.backend_url)
                async with httpx.AsyncClient(timeout=180, limits=httpx.Limits(
                    max_connections=max(10, cell.parallel * 2),
                )) as client:
                    # Warm every configuration, including controlled runs.
                    warmup = await api.play(
                        client, run_id, True, args.discussion_seconds, rounds, observations,
                    )
                    if warmup["outcome"] != "complete":
                        raise RuntimeError("Warmup failed; backend must restart before measurement")
                    for batch_index in range(args.repetitions):
                        batch = await asyncio.gather(*(api.play(
                            client, run_id, False, args.discussion_seconds, rounds, observations,
                        ) for _ in range(cell.parallel)))
                        for row in batch:
                            row["batch_index"] = batch_index
                        if any(row["outcome"] != "complete" for row in batch):
                            # There is no gameplay cancellation API. Stop this cell instead of
                            # letting failed-but-still-running rounds contaminate another batch.
                            raise RuntimeError("Measured batch failed; stopping experiment")
                    # Allow cancelled background-turn finalizers to flush their timing events.
                    await asyncio.sleep(.25)
            finally:
                if backend is not None:
                    backend.stop()
                if stub is not None:
                    stub.stop()
                events.extend(read_events(directory / "events.jsonl"))
                # Persist progress after each cell, including interrupted/failed cells.
                write_json(output / "rounds.json", rounds)
                write_json(output / "observations.json", observations)
    except Exception as exc:
        exit_error = type(exc).__name__
    finally:
        (output / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
        summary = summarize(output, rounds, events, observations, runner_error=exit_error)
        if not args.no_mlflow:
            tracker = MlflowTracker(TrackerConfig(
                "game_latency", "sqlite:///" + (output.parent / "mlflow.db").as_posix(), output,
            ))
            safe_results = [BackendExperimentResult(
                request=BackendExperimentRequest("fixed_fixture", "fixed_fixture"),
                success=row["outcome"] == "complete", round_id=row["round_id"],
                status=row["outcome"], latency_ms=row["latency_ms"],
            ) for row in rounds if not row.get("warmup")]
            try:
                tracker.track_run(run_name="tracking", results=safe_results,
                                  metrics={"measured_games": summary["measured_games"],
                                           "failed_games": summary["failed_games"],
                                           "cells": {
                                               cell: {row["metric"]: {
                                                   k: v for k, v in row.items()
                                                   if k not in {"run_id", "metric"}
                                               } for row in summary["metrics"] if row["run_id"] == cell}
                                               for cell in {row["run_id"] for row in summary["metrics"]}
                                           },
                                           "rates": {row["run_id"]: {
                                               k: v for k, v in row.items() if k != "run_id"
                                           } for row in summary["rates"]}}, params=params)
                # Attach detailed timing artifacts to the most recently created tracking run.
                import mlflow
                with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id):
                    for name in ("events.jsonl", "rounds.csv", "turns.csv", "messages.csv", "report.html"):
                        mlflow.log_artifact(str(output / name))
            except Exception as exc:
                write_json(output / "tracking_error.json", {"error_type": type(exc).__name__})
    print(f"Report: {output / 'report.html'}", flush=True)
    return 1 if exit_error or summary["failed_games"] else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
