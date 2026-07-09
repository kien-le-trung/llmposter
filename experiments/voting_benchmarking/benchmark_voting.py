from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

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
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Print per-run progress while collecting rounds and semantic features.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N completed rounds or semantic batches.",
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
            show_progress=args.show_progress,
            progress_every=args.progress_every,
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

    total_rounds = config.repetitions * len(cases)
    completed_round_count = 0
    for repetition_index in range(config.repetitions):
        for case_index, case in enumerate(cases):
            if config.show_progress:
                print(
                    "Starting round "
                    f"{completed_round_count + 1}/{total_rounds} "
                    f"(technique={config.technique}, "
                    f"repetition={repetition_index + 1}/{config.repetitions}, "
                    f"case={case_index + 1}/{len(cases)}, "
                    f"secret_word={case.secret_word!r})"
                )
            round_record, case_clues, vote_record = await client.run_case(
                case,
                config.technique,
                run_id,
            )
            round_records.append(round_record)
            clue_records.extend(case_clues)
            if vote_record is not None:
                vote_records.append(vote_record)
            completed_round_count += 1
            if _should_report_progress(config, completed_round_count, total_rounds):
                status = "ok" if round_record.success else "failed"
                print(
                    "Completed round "
                    f"{completed_round_count}/{total_rounds} "
                    f"(status={status}, latency_ms={round_record.latency_ms:.1f}, "
                    f"round_id={round_record.round_id})"
                )

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
        return await _compute_semantic_features(clues, config)
    except Exception as exc:
        if config.require_semantic_features:
            raise
        print(f"Semantic feature extraction skipped: {exc}")
        return []


async def _compute_semantic_features(
    clues: list[ClueRecord],
    config: BenchmarkConfig,
) -> list[SemanticFeatureRecord]:
    backend_path = Path(__file__).resolve().parents[2] / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    from app.services.agents.inference import InferenceClient  # noqa: PLC0415

    records: list[SemanticFeatureRecord] = []
    settings = _load_embedding_settings(backend_path)
    client = InferenceClient(settings=settings)
    clues_by_round: dict[str, list[ClueRecord]] = {}
    for clue in clues:
        clues_by_round.setdefault(clue.round_id, []).append(clue)

    total_rounds = len(clues_by_round)
    for round_index, round_clues in enumerate(clues_by_round.values(), start=1):
        if not round_clues:
            continue
        if _should_report_progress(config, round_index, total_rounds):
            print(
                "Embedding semantic batch "
                f"{round_index}/{total_rounds} "
                f"(round_id={round_clues[0].round_id}, clues={len(round_clues)})"
            )

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


def _should_report_progress(
    config: BenchmarkConfig,
    completed_count: int,
    total_count: int,
) -> bool:
    if not config.show_progress:
        return False
    return completed_count == total_count or completed_count % config.progress_every == 0


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


def _load_embedding_settings(backend_path: Path) -> SimpleNamespace:
    root_path = backend_path.parent
    return SimpleNamespace(
        embedding_model_server_url=_get_setting_value(
            "EMBEDDING_MODEL_SERVER_URL",
            "http://localhost:11434",
            backend_path,
            root_path,
        ),
        embedding_model_name=_get_setting_value(
            "EMBEDDING_MODEL_NAME",
            "nomic-embed-text",
            backend_path,
            root_path,
        ),
        inference_mode=_get_setting_value(
            "INFERENCE_MODE",
            "remote",
            backend_path,
            root_path,
        ),
        llm_config=None,
    )


def _get_setting_value(
    name: str,
    default: str,
    backend_path: Path,
    root_path: Path,
) -> str:
    value = os.getenv(name)
    if value is not None:
        return value

    for env_path in (backend_path / ".env", root_path / ".env"):
        value = _read_env_file_value(env_path, name)
        if value is not None:
            return value

    return default


def _read_env_file_value(env_path: Path, name: str) -> str | None:
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'")

    return None


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
        mlflow.log_param("show_progress", config.show_progress)
        mlflow.log_param("progress_every", config.progress_every)

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
