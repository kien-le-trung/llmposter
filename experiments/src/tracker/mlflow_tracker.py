from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from backend_handler.api_client import BackendExperimentResult


@dataclass(frozen=True)
class TrackerConfig:
    experiment_name: str
    tracking_uri: str
    output_dir: Path


@dataclass(frozen=True)
class TrackedRunArtifacts:
    run_id: str
    output_dir: Path
    raw_results_path: Path
    metrics_path: Path
    params_path: Path


class MlflowTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config

    def track_run(
        self,
        *,
        run_name: str,
        results: list[BackendExperimentResult],
        metrics: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> TrackedRunArtifacts:
        run_id = _safe_name(run_name)
        run_dir = self.config.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        raw_results_path = run_dir / "raw_results.jsonl"
        metrics_path = run_dir / "metrics.json"
        params_path = run_dir / "params.json"

        effective_params = params or {}
        _write_jsonl(raw_results_path, [result.to_dict() for result in results])
        _write_json(metrics_path, metrics)
        _write_json(params_path, effective_params)

        self._log_to_mlflow(
            run_name=run_name,
            params=effective_params,
            metrics=metrics,
            artifact_paths=[raw_results_path, metrics_path, params_path],
        )

        return TrackedRunArtifacts(
            run_id=run_id,
            output_dir=run_dir,
            raw_results_path=raw_results_path,
            metrics_path=metrics_path,
            params_path=params_path,
        )

    def _log_to_mlflow(
        self,
        *,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        artifact_paths: list[Path],
    ) -> None:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

        with mlflow.start_run(run_name=run_name):
            for name, value in _flatten_mapping(params).items():
                mlflow.log_param(name, _stringify_param(value))

            for name, value in _flatten_mapping(metrics).items():
                if isinstance(value, bool):
                    mlflow.log_metric(name, float(value))
                elif isinstance(value, (int, float)):
                    mlflow.log_metric(name, float(value))

            for artifact_path in artifact_paths:
                mlflow.log_artifact(str(artifact_path))


def build_timestamped_run_name(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, default=str) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _flatten_mapping(
    mapping: dict[str, Any],
    prefix: str | None = None,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(value, name))
        else:
            flattened[name] = value
    return flattened


def _stringify_param(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, default=str)


def _safe_name(value: str) -> str:
    cleaned = [
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value.strip()
    ]
    return "".join(cleaned).strip("_") or "run"
