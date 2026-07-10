from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_DIR = Path(__file__).resolve().parents[3]
RUNNER_DIR = Path(__file__).resolve().parent
SINGLES_DIR = RUNNER_DIR / "singles"
SWEEPS_DIR = RUNNER_DIR / "sweeps"

EXPERIMENT_TYPES = {"prompt", "voting", "combined"}


@dataclass(frozen=True)
class BackendConfig:
    host: str
    port: int
    reload: bool
    startup_timeout_seconds: float


@dataclass(frozen=True)
class ComponentConfig:
    llm: str
    embedding: str
    prompt: str
    eval_dataset: str
    voting_algo: str
    inference_mode: str
    agent_config_source: str
    word_selection_mode: str


@dataclass(frozen=True)
class RunConfig:
    repetitions: int
    timeout_seconds: float
    submit_vote: bool


@dataclass(frozen=True)
class TrackingConfig:
    experiment_name: str
    tracking_uri: str
    output_dir: Path


@dataclass(frozen=True)
class SingleExperimentConfig:
    name: str
    experiment_type: str
    backend: BackendConfig
    components: ComponentConfig
    run: RunConfig
    tracking: TrackingConfig
    source_path: Path | None = None
    sweep_name: str | None = None
    sweep_variant_index: int | None = None
    sweep_values: dict[str, Any] | None = None

    def to_params(self) -> dict[str, Any]:
        params = asdict(self)
        params["tracking"]["output_dir"] = str(self.tracking.output_dir)
        if self.source_path is not None:
            params["source_path"] = str(self.source_path)
        return params


@dataclass(frozen=True)
class SweepConfig:
    name: str
    base: dict[str, Any]
    vary: dict[str, list[Any]]
    source_path: Path | None = None


def load_single_config(relative_path: str | Path) -> SingleExperimentConfig:
    path = resolve_config_path(SINGLES_DIR, relative_path)
    data = _read_yaml_mapping(path)
    return parse_single_config(data, source_path=path)


def load_sweep_config(relative_path: str | Path) -> SweepConfig:
    path = resolve_config_path(SWEEPS_DIR, relative_path)
    data = _read_yaml_mapping(path)
    return parse_sweep_config(data, source_path=path)


def parse_single_config(
    data: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> SingleExperimentConfig:
    required = ["name", "experiment_type", "backend", "components", "run", "tracking"]
    _require_keys(data, required, "single experiment")

    experiment_type = _as_str(data["experiment_type"], "experiment_type")
    if experiment_type not in EXPERIMENT_TYPES:
        raise ValueError(
            "experiment_type must be one of "
            f"{', '.join(sorted(EXPERIMENT_TYPES))}; got {experiment_type!r}"
        )

    backend = _as_mapping(data["backend"], "backend")
    components = _as_mapping(data["components"], "components")
    run = _as_mapping(data["run"], "run")
    tracking = _as_mapping(data["tracking"], "tracking")

    return SingleExperimentConfig(
        name=_as_str(data["name"], "name"),
        experiment_type=experiment_type,
        backend=BackendConfig(
            host=_as_str(backend.get("host"), "backend.host"),
            port=_as_int(backend.get("port"), "backend.port"),
            reload=_as_bool(backend.get("reload"), "backend.reload"),
            startup_timeout_seconds=_as_float(
                backend.get("startup_timeout_seconds"),
                "backend.startup_timeout_seconds",
            ),
        ),
        components=ComponentConfig(
            llm=_as_str(components.get("llm"), "components.llm"),
            embedding=_as_str(components.get("embedding"), "components.embedding"),
            prompt=_as_str(components.get("prompt"), "components.prompt"),
            eval_dataset=_as_str(
                components.get("eval_dataset"),
                "components.eval_dataset",
            ),
            voting_algo=_as_str(components.get("voting_algo"), "components.voting_algo"),
            inference_mode=_as_str(
                components.get("inference_mode"),
                "components.inference_mode",
            ),
            agent_config_source=_as_str(
                components.get("agent_config_source"),
                "components.agent_config_source",
            ),
            word_selection_mode=_as_str(
                components.get("word_selection_mode"),
                "components.word_selection_mode",
            ),
        ),
        run=RunConfig(
            repetitions=_as_int(run.get("repetitions"), "run.repetitions"),
            timeout_seconds=_as_float(run.get("timeout_seconds"), "run.timeout_seconds"),
            submit_vote=_as_bool(run.get("submit_vote"), "run.submit_vote"),
        ),
        tracking=TrackingConfig(
            experiment_name=_as_str(
                tracking.get("experiment_name"),
                "tracking.experiment_name",
            ),
            tracking_uri=_as_str(tracking.get("tracking_uri"), "tracking.tracking_uri"),
            output_dir=_resolve_repo_path(
                _as_str(tracking.get("output_dir"), "tracking.output_dir")
            ),
        ),
        source_path=source_path,
        sweep_name=_optional_str(data.get("sweep_name"), "sweep_name"),
        sweep_variant_index=_optional_int(
            data.get("sweep_variant_index"),
            "sweep_variant_index",
        ),
        sweep_values=(
            dict(_as_mapping(data["sweep_values"], "sweep_values"))
            if "sweep_values" in data
            else None
        ),
    )


def parse_sweep_config(
    data: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> SweepConfig:
    _require_keys(data, ["name", "base", "vary"], "sweep")
    vary = _as_mapping(data["vary"], "vary")
    parsed_vary: dict[str, list[Any]] = {}
    for key, values in vary.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"vary.{key} must be a non-empty list")
        parsed_vary[str(key)] = values

    return SweepConfig(
        name=_as_str(data["name"], "name"),
        base=dict(_as_mapping(data["base"], "base")),
        vary=parsed_vary,
        source_path=source_path,
    )


def resolve_config_path(base_dir: Path, relative_path: str | Path) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        candidate = path
    else:
        candidate = base_dir / path
    if not candidate.exists():
        raise FileNotFoundError(f"Config file not found: {candidate}")
    return candidate.resolve()


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return dict(_as_mapping(data, str(path)))


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_DIR / candidate).resolve()


def _require_keys(data: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing required {context} keys: {', '.join(missing)}")


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _as_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _as_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _as_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, field_name)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field_name)
