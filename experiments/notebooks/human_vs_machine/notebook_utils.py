from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "experiments" / "datasets" / "human_behavior_manifest.json").exists():
            return candidate
    raise FileNotFoundError("Could not locate the llmposter repository root.")


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int, repetitions: int = 10_000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(repetitions, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def paired_sign_permutation_pvalue(
    differences: Sequence[float], *, seed: int, repetitions: int = 20_000
) -> float:
    array = np.asarray(differences, dtype=float)
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(repetitions, array.size))
    permuted = np.abs((signs * array).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (repetitions + 1))


def standardized_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    standard_deviation = array.std(ddof=1)
    return float(array.mean() / standard_deviation) if standard_deviation else 0.0


def result_record(
    *,
    metric: str,
    values: Sequence[float],
    seed: int,
    test: str,
    primary: bool,
    p_value: float | None = None,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    low, high = bootstrap_mean_ci(array, seed=seed)
    return {
        "metric": metric,
        "sample_size_rounds": int(array.size),
        "estimate": float(array.mean()),
        "effect_size": standardized_mean(array),
        "ci_95_low": low,
        "ci_95_high": high,
        "p_value": p_value,
        "test": test,
        "seed": seed,
        "primary": primary,
    }


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 1.0
    return float(1.0 - np.dot(left, right) / denominator)


def ollama_embeddings(
    texts: Sequence[str],
    *,
    cache_path: Path,
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    timeout_seconds: float = 30.0,
) -> np.ndarray | None:
    normalized = [" ".join(text.casefold().split()) for text in texts]
    cache_key = hashlib.sha256(
        json.dumps({"model": model, "texts": normalized}).encode("utf-8")
    ).hexdigest()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key:
            return np.asarray(cached["embeddings"], dtype=float)

    payload = json.dumps({"model": model, "input": normalized}).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            data = json.load(response)
    except (OSError, URLError, TimeoutError) as exc:
        print(f"Skipping optional Ollama analysis: {exc}")
        return None

    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(normalized):
        print("Skipping optional Ollama analysis: unexpected response shape")
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"cache_key": cache_key, "model": model, "embeddings": embeddings}),
        encoding="utf-8",
    )
    return np.asarray(embeddings, dtype=float)


def cluster_bootstrap_difference(
    round_values: Sequence[float], *, seed: int, repetitions: int = 10_000
) -> tuple[float, float, float]:
    array = np.asarray(round_values, dtype=float)
    low, high = bootstrap_mean_ci(array, seed=seed, repetitions=repetitions)
    return float(array.mean()), low, high


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
