from __future__ import annotations

from typing import Any

from backend_handler.api_client import (
    BackendExperimentApiClient,
    BackendExperimentRequest,
    BackendExperimentResult,
)
from evaluator import evaluate_common, evaluate_prompt, evaluate_voting
from tracker import MlflowTracker, TrackerConfig, TrackedRunArtifacts
from tracker.mlflow_tracker import build_timestamped_run_name

from .backend_process import ManagedBackend
from .config import SingleExperimentConfig
from .dataset_loader import DatasetInfo, ExperimentCase, load_dataset


async def run_single_experiment(
    config: SingleExperimentConfig,
) -> TrackedRunArtifacts:
    dataset_info, cases = load_dataset(config.components.eval_dataset)
    run_name = build_timestamped_run_name(config.name)
    tracker = MlflowTracker(
        TrackerConfig(
            experiment_name=config.tracking.experiment_name,
            tracking_uri=config.tracking.tracking_uri,
            output_dir=config.tracking.output_dir,
        )
    )
    params = _build_params(config, dataset_info, len(cases), None)

    results: list[BackendExperimentResult] = []
    metrics: dict[str, Any] = {}
    try:
        async with ManagedBackend(config.backend, config.components) as backend:
            client = BackendExperimentApiClient(
                backend.backend_url,
                timeout_seconds=config.run.timeout_seconds,
            )
            results = await _run_requests(client, config, cases)
            metrics = evaluate_results(config.experiment_type, results)
            params["backend_url"] = backend.backend_url
    except Exception as exc:
        metrics = evaluate_common(results)
        metrics["runner_failed"] = True
        metrics["runner_error"] = str(exc) or exc.__class__.__name__
        params["runner_error"] = metrics["runner_error"]
        artifacts = tracker.track_run(
            run_name=run_name,
            results=results,
            metrics=metrics,
            params=params,
        )
        raise RuntimeError(f"Experiment {config.name} failed: {metrics['runner_error']}") from exc

    return tracker.track_run(
        run_name=run_name,
        results=results,
        metrics=metrics,
        params=params,
    )


async def _run_requests(
    client: BackendExperimentApiClient,
    config: SingleExperimentConfig,
    cases: list[ExperimentCase],
) -> list[BackendExperimentResult]:
    results: list[BackendExperimentResult] = []
    for repetition_index in range(config.run.repetitions):
        for case in cases:
            request = BackendExperimentRequest(
                secret_word=case.secret_word,
                imposter_hint=case.imposter_hint,
                prompt_technique=config.components.prompt,
                include_human=False,
                submit_vote=config.run.submit_vote,
                case_id=case.case_id,
                repetition_index=repetition_index,
            )
            results.append(await client.run_request(request))
    return results


def evaluate_results(
    experiment_type: str,
    results: list[BackendExperimentResult],
) -> dict[str, Any]:
    metrics = evaluate_common(results)
    if experiment_type in {"prompt", "voting", "combined"}:
        metrics.update(evaluate_prompt(results))
    if experiment_type in {"voting", "combined"}:
        metrics.update(evaluate_voting(results))
    return metrics


def _build_params(
    config: SingleExperimentConfig,
    dataset_info: DatasetInfo,
    case_count: int,
    backend_url: str | None,
) -> dict[str, Any]:
    return {
        "config": config.to_params(),
        "components": config.to_params()["components"],
        "dataset": dataset_info.to_dict(),
        "case_count": case_count,
        "repetitions": config.run.repetitions,
        "backend_url": backend_url,
        "sweep_name": config.sweep_name,
        "sweep_variant_index": config.sweep_variant_index,
        "sweep_values": config.sweep_values,
    }
