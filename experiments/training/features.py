from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import Pipeline
except ImportError as exc:  # pragma: no cover - exercised only without sklearn.
    raise ImportError(
        "features.py requires scikit-learn. Install scikit-learn before running "
        "feature generation."
    ) from exc


REPO_DIR = Path(__file__).resolve().parents[2]
TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = TRAINING_DIR / "datasets" / "scraped"
DEFAULT_OUTPUT_DIR = TRAINING_DIR / "datasets" / "featured"
DEFAULT_EMBEDDING_CONFIG = (
    REPO_DIR / "experiments" / "run_configs" / "embedding_configs" / "nomic_embed_text.json"
)

INPUT_PATTERN = "voting_candidates_*.csv"
INPUT_PREFIX = "voting_candidates_"
OUTPUT_PREFIX = "voting_features_"

FEATURE_SCHEMA_VERSION = 1
FEATURE_BUILDER_NAME = "embedding_similarity_v1"
FEATURE_COLUMNS = [
    "candidate_turn_position",
    "embedding_similarity_to_others_mean",
    "embedding_similarity_to_others_std",
    "embedding_similarity_rank_low_to_high",
    "embedding_similarity_to_previous_mean",
]

PRESERVED_COLUMNS = [
    "experiment_name",
    "case_id",
    "round_id",
    "turn_id",
    "secret_word",
    "imposter_was",
    "candidate_agent_id",
    "candidate_agent_name",
    "candidate_clue",
    "is_imposter",
    "all_clues_json",
]


@dataclass(frozen=True)
class FeatureBuildResult:
    source_path: Path
    output_path: Path
    manifest_path: Path
    row_count: int


class OllamaEmbeddingClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        api_request = request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(api_request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not reach embedding server at {self.base_url}. "
                "Start the local embedding model on port 11434 or pass --base-url."
            ) from exc

        embedding = response_payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(
                f"Embedding server returned no embedding for model {self.model!r}."
            )

        return [float(value) for value in embedding]


class VotingFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: list[dict[str, Any]], y: Any = None) -> VotingFeatureTransformer:
        return self

    def transform(self, X: list[dict[str, Any]]) -> list[dict[str, float]]:
        grouped = _group_indices_by_round(X)
        similarity_means_by_index: dict[int, float] = {}

        for indices in grouped.values():
            for index in indices:
                similarities = _candidate_to_other_similarities(X[index])
                similarity_means_by_index[index] = _mean(similarities)

        ranks_by_index = _rank_similarity_means(grouped, similarity_means_by_index)

        features: list[dict[str, float]] = []
        for index, record in enumerate(X):
            similarities = _candidate_to_other_similarities(record)
            previous_similarities = _candidate_to_previous_similarities(record)
            features.append(
                {
                    "candidate_turn_position": float(record["candidate_turn_position"]),
                    "embedding_similarity_to_others_mean": _mean(similarities),
                    "embedding_similarity_to_others_std": _std(similarities),
                    "embedding_similarity_rank_low_to_high": float(ranks_by_index[index]),
                    "embedding_similarity_to_previous_mean": _mean(previous_similarities),
                }
            )

        return features


class FeatureBuilder:
    def __init__(
        self,
        *,
        embedding_client: OllamaEmbeddingClient,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.embedding_client = embedding_client
        self.output_dir = output_dir
        self.pipeline = Pipeline(
            [("voting_features", VotingFeatureTransformer())]
        )

    def build_file(self, source_path: Path) -> FeatureBuildResult:
        rows = _read_csv(source_path)
        intermediate_rows = self._build_intermediate_rows(rows)
        feature_rows = self.pipeline.fit_transform(intermediate_rows)

        dataset_version = _dataset_version_from_source(source_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{OUTPUT_PREFIX}{dataset_version}.csv"
        manifest_path = self.output_dir / f"{OUTPUT_PREFIX}{dataset_version}.manifest.json"

        output_rows = [
            _featured_output_row(source=row, features=features)
            for row, features in zip(rows, feature_rows, strict=True)
        ]
        _write_csv(output_path, output_rows)
        _write_manifest(
            manifest_path,
            {
                "dataset_version": dataset_version,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "row_count": len(output_rows),
                "embedding_model": self.embedding_client.model,
                "embedding_base_url": self.embedding_client.base_url,
                "feature_columns": FEATURE_COLUMNS,
                "preserved_columns": [
                    column
                    for column in PRESERVED_COLUMNS
                    if column in rows[0] and column != "repetition_index"
                ]
                if rows
                else [],
                "pipeline_steps": [name for name, _ in self.pipeline.steps],
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

        return FeatureBuildResult(
            source_path=source_path,
            output_path=output_path,
            manifest_path=manifest_path,
            row_count=len(output_rows),
        )

    def _build_intermediate_rows(
        self,
        rows: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        embedding_cache = self._embed_unique_clues(rows)
        return [
            _intermediate_row(row, embedding_cache)
            for row in rows
        ]

    def _embed_unique_clues(self, rows: list[dict[str, str]]) -> dict[str, list[float]]:
        unique_clues: set[str] = set()
        for row in rows:
            for clue in _parse_all_clues(row["all_clues_json"]):
                unique_clues.add(clue["clue"])

        return {
            clue: self.embedding_client.embed(clue)
            for clue in sorted(unique_clues)
        }


def parse_args() -> argparse.Namespace:
    config = _load_embedding_config(DEFAULT_EMBEDDING_CONFIG)
    parser = argparse.ArgumentParser(
        description="Build production-like voting features from scraped candidate CSVs."
    )
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        default=config.get("embedding_model_name", "nomic-embed-text"),
    )
    parser.add_argument(
        "--base-url",
        default=config.get("embedding_model_server_url", "http://localhost:11434"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = _resolve_source_paths(args.input_file, args.input_dir)
    client = OllamaEmbeddingClient(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    builder = FeatureBuilder(embedding_client=client, output_dir=args.output_dir)

    for source_path in source_paths:
        result = builder.build_file(source_path)
        print(f"Wrote {result.output_path}")
        print(f"Wrote {result.manifest_path}")


def _resolve_source_paths(input_file: Path | None, input_dir: Path) -> list[Path]:
    if input_file is not None:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        return [input_file]

    paths = sorted(input_dir.glob(INPUT_PATTERN))
    if not paths:
        raise FileNotFoundError(
            f"No scraped datasets found at {input_dir / INPUT_PATTERN}."
        )
    return paths


def _load_embedding_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = _output_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _output_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return [
            column
            for column in PRESERVED_COLUMNS
            if column != "repetition_index"
        ] + FEATURE_COLUMNS

    preserved = [
        column
        for column in PRESERVED_COLUMNS
        if column in rows[0] and column != "repetition_index"
    ]
    return preserved + FEATURE_COLUMNS


def _featured_output_row(
    *,
    source: dict[str, str],
    features: dict[str, float],
) -> dict[str, Any]:
    output = {
        column: source.get(column, "")
        for column in PRESERVED_COLUMNS
        if column in source and column != "repetition_index"
    }
    output.update(features)
    return output


def _intermediate_row(
    row: dict[str, str],
    embedding_cache: dict[str, list[float]],
) -> dict[str, Any]:
    all_clues = _parse_all_clues(row["all_clues_json"])
    candidate_index = _candidate_index(row, all_clues)
    candidate_clue = all_clues[candidate_index]["clue"]

    return {
        "round_id": row.get("round_id", ""),
        "turn_id": row.get("turn_id", ""),
        "candidate_turn_position": candidate_index,
        "candidate_embedding": embedding_cache[candidate_clue],
        "other_embeddings": [
            embedding_cache[clue["clue"]]
            for index, clue in enumerate(all_clues)
            if index != candidate_index
        ],
        "previous_embeddings": [
            embedding_cache[clue["clue"]]
            for clue in all_clues[:candidate_index]
        ],
    }


def _parse_all_clues(raw_value: str) -> list[dict[str, str]]:
    payload = json.loads(raw_value)
    if not isinstance(payload, list) or not payload:
        raise ValueError("all_clues_json must be a non-empty JSON array.")

    clues: list[dict[str, str]] = []
    for clue in payload:
        if not isinstance(clue, dict):
            continue
        clues.append(
            {
                "agent_id": _string_value(clue.get("agent_id")),
                "agent_name": _string_value(clue.get("agent_name")),
                "clue": _string_value(clue.get("clue")),
            }
        )

    if not clues:
        raise ValueError("all_clues_json did not contain any usable clues.")
    return clues


def _candidate_index(row: dict[str, str], all_clues: list[dict[str, str]]) -> int:
    candidate_agent_id = row.get("candidate_agent_id", "")
    candidate_clue = row.get("candidate_clue", "")

    for index, clue in enumerate(all_clues):
        if clue["agent_id"] == candidate_agent_id:
            return index

    for index, clue in enumerate(all_clues):
        if clue["clue"] == candidate_clue:
            return index

    raise ValueError(
        "Could not match candidate row to all_clues_json by agent_id or clue."
    )


def _candidate_to_other_similarities(record: dict[str, Any]) -> list[float]:
    return [
        _cosine_similarity(record["candidate_embedding"], embedding)
        for embedding in record["other_embeddings"]
    ]


def _candidate_to_previous_similarities(record: dict[str, Any]) -> list[float]:
    return [
        _cosine_similarity(record["candidate_embedding"], embedding)
        for embedding in record["previous_embeddings"]
    ]


def _group_indices_by_round(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        key = (str(record["round_id"]), str(record["turn_id"]))
        grouped.setdefault(key, []).append(index)
    return grouped


def _rank_similarity_means(
    grouped: dict[tuple[str, str], list[int]],
    means_by_index: dict[int, float],
) -> dict[int, int]:
    ranks: dict[int, int] = {}
    for indices in grouped.values():
        sorted_indices = sorted(
            indices,
            key=lambda index: (
                means_by_index[index],
                index,
            ),
        )
        for rank, index in enumerate(sorted_indices):
            ranks[index] = rank
    return ranks


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compare embeddings with different dimensions.")

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _dataset_version_from_source(path: Path) -> str:
    stem = path.stem
    if not stem.startswith(INPUT_PREFIX):
        raise ValueError(f"Expected input filename to start with {INPUT_PREFIX!r}: {path}")
    return stem.removeprefix(INPUT_PREFIX)


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)


if __name__ == "__main__":
    main()
