from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path

import mlflow

from api_client import VotingBenchmarkApiClient
from metrics import summarize_results
from schemas import (
    BenchmarkCase,
    BenchmarkConfig,
    ClueRecord,
    RoundArtifactRecord,
    SemanticFeatureRecord,
    VoteRecord,
)

SUPPORTED_TECHNIQUES = {"zero_shot", "few_shot", "reasoning_guided", "meta"}
MLFLOW_DB_PATH = "experiments/voting_benchmarking/mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("voting_benchmarking")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect replayable voting benchmark artifacts and difficulty metrics."
    )
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument(
        "--cases",
        default="experiments/voting_benchmarking/benchmark_cases.json",
    )
    parser.add_argument(
        "--technique",
        default="few_shot",
        choices=[*sorted(SUPPORTED_TECHNIQUES), "all"],
    )
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="experiments/voting_benchmarking/results",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--skip-semantic-features",
        action="store_true",
        help="Collect round and vote artifacts without embedding-based semantic metrics.",
    )
    parser.add_argument(
        "--require-semantic-features",
        action="store_true",
        help="Fail the run if semantic feature extraction fails.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = _load_cases(Path(args.cases))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    client = VotingBenchmarkApiClient(
        backend_url=args.backend_url,
        timeout_seconds=args.timeout_seconds,
    )
    techniques = (
        sorted(SUPPORTED_TECHNIQUES)
        if args.technique == "all"
        else [args.technique]
    )

    for technique in techniques:
        config = BenchmarkConfig(
            backend_url=args.backend_url,
            technique=technique,
            repetitions=args.repetitions,
            timeout_seconds=args.timeout_seconds,
            compute_semantic_features=not args.skip_semantic_features,
            require_semantic_features=args.require_semantic_features,
        )
        await _run_technique_benchmark(
            client=client,
            config=config,
            cases=cases,
            output_dir=output_dir,
            timestamp=timestamp,
        )


async def _run_technique_benchmark(
    client: VotingBenchmarkApiClient,
    config: BenchmarkConfig,
    cases: list[BenchmarkCase],
    output_dir: Path,
    timestamp: str,
) -> None:
    run_id = f"{config.technique}_{timestamp}"
    print(
        f"Running voting benchmark {run_id} against {config.backend_url}. "
        "Current backend voting is recorded as embedding_distance_v1."
    )
    round_records: list[RoundArtifactRecord] = []
    clue_records: list[ClueRecord] = []
    vote_records: list[VoteRecord] = []

    for _ in range(config.repetitions):
        for case in cases:
            round_record, case_clues, vote_record = await client.run_case(
                case,
                config.technique,
                run_id,
            )
            round_records.append(round_record)
            clue_records.extend(case_clues)
            if vote_record is not None:
                vote_records.append(vote_record)

    semantic_features = await _build_semantic_features(
        config,
        clue_records,
    )
    summary = summarize_results(
        config.technique,
        round_records,
        vote_records,
        semantic_features,
    )

    rounds_path = output_dir / f"round_artifacts_{config.technique}_{timestamp}.jsonl"
    clues_path = output_dir / f"clues_{config.technique}_{timestamp}.jsonl"
    votes_path = output_dir / f"votes_{config.technique}_{timestamp}.jsonl"
    semantic_path = output_dir / f"semantic_features_{config.technique}_{timestamp}.jsonl"
    summary_path = output_dir / f"summary_{config.technique}_{timestamp}.json"

    _write_jsonl(rounds_path, [record.model_dump() for record in round_records])
    _write_jsonl(clues_path, [record.model_dump() for record in clue_records])
    _write_jsonl(votes_path, [record.model_dump() for record in vote_records])
    _write_jsonl(semantic_path, [record.model_dump() for record in semantic_features])
    summary_path.write_text(
        json.dumps(summary.model_dump(), indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary.model_dump(), indent=2))
    _log_mlflow_run(
        run_id,
        config,
        cases,
        summary.model_dump(),
        [rounds_path, clues_path, votes_path, semantic_path, summary_path],
    )


async def _build_semantic_features(
    config: BenchmarkConfig,
    clues: list[ClueRecord],
) -> list[SemanticFeatureRecord]:
    if not config.compute_semantic_features or not clues:
        return []

    try:
        return await _compute_semantic_features(clues)
    except Exception as exc:
        if config.require_semantic_features:
            raise
        print(f"Semantic feature extraction skipped: {exc}")
        return []


async def _compute_semantic_features(clues: list[ClueRecord]) -> list[SemanticFeatureRecord]:
    backend_path = Path(__file__).resolve().parents[2] / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    from app.core.config import settings  # noqa: PLC0415
    from app.services.agents.inference import InferenceClient  # noqa: PLC0415

    records: list[SemanticFeatureRecord] = []
    client = InferenceClient(settings=settings)
    clues_by_round: dict[str, list[ClueRecord]] = {}
    for clue in clues:
        clues_by_round.setdefault(clue.round_id, []).append(clue)

    for round_clues in clues_by_round.values():
        if not round_clues:
            continue

        first_clue = round_clues[0]
        texts = [
            first_clue.secret_word,
            first_clue.imposter_hint,
            *[clue.clue for clue in round_clues],
        ]
        embedding_result = await client.embed(texts)
        secret_embedding = embedding_result.embeddings[0]
        hint_embedding = embedding_result.embeddings[1]
        clue_embeddings = dict(
            zip(
                [clue.player_id for clue in round_clues],
                embedding_result.embeddings[2:],
                strict=True,
            )
        )
        non_imposter_clues = [
            clue for clue in round_clues if clue.role == "non_imposter"
        ]
        non_imposter_embeddings = [
            clue_embeddings[clue.player_id] for clue in non_imposter_clues
        ]
        non_imposter_centroid = (
            _centroid(non_imposter_embeddings)
            if non_imposter_embeddings
            else None
        )
        pairwise_similarity = _mean_pairwise_similarity(non_imposter_embeddings)
        non_imposter_distances = [
            1.0 - _cosine_similarity(embedding, non_imposter_centroid)
            for embedding in non_imposter_embeddings
            if non_imposter_centroid is not None
        ]
        mean_non_imposter_distance = (
            sum(non_imposter_distances) / len(non_imposter_distances)
            if non_imposter_distances
            else None
        )

        for clue in round_clues:
            clue_embedding = clue_embeddings[clue.player_id]
            clue_centroid_similarity = (
                _cosine_similarity(clue_embedding, non_imposter_centroid)
                if non_imposter_centroid is not None
                else None
            )
            imposter_outlier_score = None
            separability_margin = None
            if clue.role == "imposter" and non_imposter_centroid is not None:
                imposter_distance = 1.0 - _cosine_similarity(
                    clue_embedding,
                    non_imposter_centroid,
                )
                imposter_outlier_score = imposter_distance
                if mean_non_imposter_distance is not None:
                    separability_margin = imposter_distance - mean_non_imposter_distance

            records.append(
                SemanticFeatureRecord(
                    run_id=clue.run_id,
                    technique=clue.technique,
                    round_id=clue.round_id,
                    player_id=clue.player_id,
                    player_name=clue.player_name,
                    role=clue.role,
                    embedding_model=settings.embedding_model_name,
                    embedding_inference_mode=embedding_result.inference_mode,
                    clue_to_secret_similarity=_cosine_similarity(
                        clue_embedding,
                        secret_embedding,
                    ),
                    clue_to_hint_similarity=_cosine_similarity(
                        clue_embedding,
                        hint_embedding,
                    ),
                    clue_to_non_imposter_centroid_similarity=clue_centroid_similarity,
                    non_imposter_pairwise_similarity=(
                        pairwise_similarity if clue.role == "imposter" else None
                    ),
                    imposter_outlier_score=imposter_outlier_score,
                    separability_margin=separability_margin,
                    hint_to_secret_similarity=_cosine_similarity(
                        hint_embedding,
                        secret_embedding,
                    ),
                )
            )

    return records


def _centroid(vectors: list[list[float]]) -> list[float]:
    dimension = len(vectors[0])
    return [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimension)
    ]


def _mean_pairwise_similarity(vectors: list[list[float]]) -> float | None:
    if len(vectors) < 2:
        return None
    similarities: list[float] = []
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            similarities.append(_cosine_similarity(left, right))
    return sum(similarities) / len(similarities)


def _cosine_similarity(left: list[float], right: list[float] | None) -> float:
    if right is None:
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def _load_cases(path: Path) -> list[BenchmarkCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkCase.model_validate(case) for case in raw_cases]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _log_mlflow_run(
    run_id: str,
    config: BenchmarkConfig,
    cases: list[BenchmarkCase],
    summary: dict,
    artifact_paths: list[Path],
) -> None:
    with mlflow.start_run(run_name=f"benchmark_{run_id}"):
        mlflow.log_param("technique", config.technique)
        mlflow.log_param("repetitions", config.repetitions)
        mlflow.log_param("case_count", len(cases))
        mlflow.log_param("backend_url", config.backend_url)
        mlflow.log_param("compute_semantic_features", config.compute_semantic_features)
        mlflow.log_param("require_semantic_features", config.require_semantic_features)

        for metric_name, metric_value in summary.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, float(metric_value))
            elif metric_name == "detection_rate_by_imposter_position":
                for position, rate in metric_value.items():
                    mlflow.log_metric(f"detection_rate_position_{position}", float(rate))

        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))


if __name__ == "__main__":
    asyncio.run(main())
