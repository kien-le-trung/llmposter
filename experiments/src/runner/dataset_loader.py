from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_DIR / "experiments"
EVAL_DATASET_CONFIGS_DIR = EXPERIMENTS_DIR / "run_configs" / "eval_dataset_configs"


@dataclass(frozen=True)
class ExperimentCase:
    secret_word: str
    imposter_hint: str
    case_id: str


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    path: Path
    description: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "description": self.description,
        }


def load_dataset(eval_dataset: str) -> tuple[DatasetInfo, list[ExperimentCase]]:
    config_path = resolve_eval_dataset_config_path(eval_dataset)
    config = _read_json_mapping(config_path)
    raw_dataset = config.get("eval_dataset")
    if not isinstance(raw_dataset, dict):
        raise ValueError(f"{config_path} must contain an eval_dataset object")

    dataset_path = _resolve_repo_path(_as_str(raw_dataset.get("path"), "eval_dataset.path"))
    dataset_info = DatasetInfo(
        name=_as_str(raw_dataset.get("name"), "eval_dataset.name"),
        path=dataset_path,
        description=(
            str(raw_dataset["description"])
            if raw_dataset.get("description") is not None
            else None
        ),
    )
    return dataset_info, load_cases(dataset_path)


def resolve_eval_dataset_config_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")

    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            Path.cwd() / path,
            REPO_DIR / path,
            EVAL_DATASET_CONFIGS_DIR / path.name,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"eval_dataset config {name_or_path!r} not found. Searched: {searched}"
    )


def load_cases(dataset_path: Path) -> list[ExperimentCase]:
    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError(f"{dataset_path} must contain a JSON array")

    cases: list[ExperimentCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"{dataset_path} case {index} must be an object")
        cases.append(
            ExperimentCase(
                secret_word=_as_str(raw_case.get("secret_word"), f"case {index}.secret_word"),
                imposter_hint=_as_str(
                    raw_case.get("imposter_hint"),
                    f"case {index}.imposter_hint",
                ),
                case_id=str(raw_case.get("id", index)),
            )
        )
    return cases


def _read_json_mapping(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_DIR / candidate).resolve()


def _as_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
