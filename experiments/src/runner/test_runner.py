from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend_handler.api_client import BackendExperimentResult  # noqa: E402
from runner.config import load_single_config, parse_single_config, parse_sweep_config  # noqa: E402
from runner.dataset_loader import load_dataset  # noqa: E402
from runner.execution import evaluate_results, run_single_experiment  # noqa: E402
from runner.sweep import expand_sweep  # noqa: E402


def test_load_single_config_resolves_relative_yaml() -> None:
    config = load_single_config("prompt_openrouter_few_shot.yaml")

    assert config.name == "prompt_openrouter_few_shot"
    assert config.experiment_type == "prompt"
    assert config.components.prompt == "few_shot"
    assert config.tracking.output_dir.is_absolute()


def test_parse_single_config_requires_top_level_keys() -> None:
    with pytest.raises(ValueError, match="Missing required single experiment keys"):
        parse_single_config({"name": "missing_sections"})


def test_expand_sweep_cartesian_grid_and_deep_apply() -> None:
    sweep = parse_sweep_config(
        {
            "name": "grid",
            "base": _single_config_dict(),
            "vary": {
                "components.prompt": ["zero_shot", "few_shot"],
                "components.llm": ["openrouter", "qwen_0_5b"],
            },
        }
    )

    variants = expand_sweep(sweep)

    assert len(variants) == 4
    assert variants[0].components.prompt == "zero_shot"
    assert variants[0].components.llm == "openrouter"
    assert variants[3].components.prompt == "few_shot"
    assert variants[3].components.llm == "qwen_0_5b"
    assert variants[3].sweep_variant_index == 3
    assert variants[3].sweep_values == {
        "components.prompt": "few_shot",
        "components.llm": "qwen_0_5b",
    }


def test_load_dataset_from_eval_dataset_config() -> None:
    dataset_info, cases = load_dataset("standard_benchmark_short")

    assert dataset_info.name == "standard_benchmark_short"
    assert dataset_info.path.name == "standard_benchmark_short.json"
    assert cases
    assert cases[0].secret_word
    assert cases[0].imposter_hint


def test_evaluate_results_selects_prompt_metrics() -> None:
    result = _result()

    metrics = evaluate_results("prompt", [result])

    assert "success_rate" in metrics
    assert "average_clue_word_count" in metrics
    assert "group_detection_rate" not in metrics


def test_evaluate_results_selects_voting_metrics() -> None:
    result = _result()

    metrics = evaluate_results("voting", [result])

    assert "success_rate" in metrics
    assert "average_clue_word_count" in metrics
    assert "group_detection_rate" in metrics


def test_run_single_experiment_orchestrates_dependencies(monkeypatch, tmp_path) -> None:
    config = parse_single_config(_single_config_dict(output_dir=str(tmp_path)))
    tracked_calls = []

    class FakeBackend:
        backend_url = "http://127.0.0.1:8000"

    class FakeManagedBackend:
        def __init__(self, backend_config, components) -> None:
            self.backend_config = backend_config
            self.components = components

        async def __aenter__(self):
            return FakeBackend()

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    class FakeClient:
        def __init__(self, backend_url, timeout_seconds) -> None:
            self.backend_url = backend_url
            self.timeout_seconds = timeout_seconds

        async def run_request(self, request):
            return _result(case_id=request.case_id, repetition_index=request.repetition_index)

    class FakeTracker:
        def __init__(self, tracker_config) -> None:
            self.tracker_config = tracker_config

        def track_run(self, *, run_name, results, metrics, params):
            tracked_calls.append(
                {
                    "run_name": run_name,
                    "results": results,
                    "metrics": metrics,
                    "params": params,
                }
            )
            return SimpleNamespace(output_dir=tmp_path)

    monkeypatch.setattr("runner.execution.ManagedBackend", FakeManagedBackend)
    monkeypatch.setattr("runner.execution.BackendExperimentApiClient", FakeClient)
    monkeypatch.setattr("runner.execution.MlflowTracker", FakeTracker)

    artifacts = asyncio.run(run_single_experiment(config))

    assert artifacts.output_dir == tmp_path
    assert len(tracked_calls) == 1
    assert len(tracked_calls[0]["results"]) == 2
    assert tracked_calls[0]["metrics"]["success_rate"] == 1.0
    assert tracked_calls[0]["params"]["backend_url"] == "http://127.0.0.1:8000"


def _single_config_dict(output_dir: str = "experiments/prompt_benchmarking/results"):
    return {
        "name": "unit_prompt",
        "experiment_type": "prompt",
        "backend": {
            "host": "127.0.0.1",
            "port": 8000,
            "reload": False,
            "startup_timeout_seconds": 30,
        },
        "components": {
            "llm": "openrouter",
            "embedding": "nomic_embed_text",
            "prompt": "few_shot",
            "eval_dataset": "standard_benchmark_short",
            "voting_algo": "embedding_distance_v1",
            "inference_mode": "remote",
            "agent_config_source": "static",
            "word_selection_mode": "random",
        },
        "run": {
            "repetitions": 1,
            "timeout_seconds": 60,
            "submit_vote": False,
        },
        "tracking": {
            "experiment_name": "prompt_benchmarking",
            "tracking_uri": "sqlite:///experiments/prompt_benchmarking/mlflow.db",
            "output_dir": output_dir,
        },
    }


def _result(case_id: str | None = "0", repetition_index: int | None = 0):
    request = SimpleNamespace(
        secret_word="orbit",
        imposter_hint="space",
        case_id=case_id,
        repetition_index=repetition_index,
        to_dict=lambda: {
            "secret_word": "orbit",
            "imposter_hint": "space",
            "case_id": case_id,
            "repetition_index": repetition_index,
        },
    )
    return BackendExperimentResult(
        request=request,
        success=True,
        round_id="round-1",
        status="ready_to_vote",
        latency_ms=12.0,
        round_payload={
            "playing_order": [
                {"id": "a1", "name": "A1", "kind": "agent"},
                {"id": "a2", "name": "A2", "kind": "agent"},
            ],
            "turns": [
                {
                    "responses": [
                        {
                            "agent_id": "a1",
                            "agent_name": "A1",
                            "agent_response": "moon path",
                        },
                        {
                            "agent_id": "a2",
                            "agent_name": "A2",
                            "agent_response": "space hint",
                        },
                    ]
                }
            ],
        },
        vote_payload={
            "imposter_was": "A2",
            "group_voted_player_name": "A2",
            "imposter_won": False,
            "agent_votes": [
                {"voter_agent_id": "a1", "voted_for": "A2"},
                {"voter_agent_id": "a2", "voted_for": "A2"},
            ],
        },
    )
