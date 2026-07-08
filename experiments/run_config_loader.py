from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_DIR / "backend"
RUN_CONFIGS_DIR = Path(__file__).resolve().parent / "run_configs"


def ensure_backend_import_path() -> None:
    backend_path = str(BACKEND_DIR)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)


def resolve_run_config_path(config_name_or_path: str) -> Path:
    config_path = Path(config_name_or_path)
    if config_path.suffix != ".json":
        config_path = config_path.with_suffix(".json")

    candidates = []
    if config_path.is_absolute():
        candidates.append(config_path)
    else:
        candidates.extend(
            [
                Path.cwd() / config_path,
                REPO_DIR / config_path,
                RUN_CONFIGS_DIR / config_path.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Run config {config_name_or_path!r} not found. Searched: {searched}")


def load_run_config_data(config_name_or_path: str) -> dict:
    config_path = resolve_run_config_path(config_name_or_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Run config {config_path} must contain a JSON object")

    return data


def load_run_settings(config_name_or_path: str):
    ensure_backend_import_path()

    from app.core.config import Settings  # noqa: PLC0415

    return Settings(**load_run_config_data(config_name_or_path))
