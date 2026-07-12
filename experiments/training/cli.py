from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG_DIR = TRAINING_DIR / "model_configs"

from .artifact import (
    ArtifactConfig,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_MLFLOW_EXPERIMENT_NAME,
    DEFAULT_MLFLOW_TRACKING_URI,
    ModelArtifact,
    ModelArtifactManager,
)
from .features import (
    DEFAULT_EMBEDDING_CONFIG,
    DEFAULT_INPUT_DIR as DEFAULT_SCRAPED_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_FEATURED_DIR,
    FeatureBuilder,
    OllamaEmbeddingClient,
    _load_embedding_config,
)
from .scrape import DEFAULT_OUTPUT_DIR as DEFAULT_SCRAPE_OUTPUT_DIR, RawResultScraper
from .train import ModelConfig, TrainingConfig, VotingModelTrainer, _resolve_dataset_paths


def parse_args() -> argparse.Namespace:
    embedding_config = _load_embedding_config(DEFAULT_EMBEDDING_CONFIG)
    parser = argparse.ArgumentParser(
        description="Orchestrate voting model training workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    full = subparsers.add_parser(
        "full",
        help="scrape raw results, build features, train, and save artifact",
    )
    full.add_argument("experiment_name")
    _add_dataset_version_arg(full)
    _add_embedding_args(full, embedding_config)
    _add_model_config_arg(full)
    _add_training_args(full)
    _add_artifact_args(full)

    scrape = subparsers.add_parser(
        "scrape",
        help="scrape raw_results.jsonl into datasets/scraped",
    )
    scrape.add_argument("experiment_name")
    _add_dataset_version_arg(scrape)
    scrape.add_argument("--output-root", type=Path, default=None)
    scrape.add_argument("--scraped-dir", type=Path, default=DEFAULT_SCRAPE_OUTPUT_DIR)

    features = subparsers.add_parser(
        "features",
        help="build featured datasets from scraped datasets",
    )
    features.add_argument("--input-file", type=Path, default=None)
    features.add_argument("--input-dir", type=Path, default=DEFAULT_SCRAPED_DIR)
    features.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURED_DIR)
    _add_embedding_args(features, embedding_config)

    train_artifact = subparsers.add_parser(
        "train-artifact",
        help="train from featured datasets and save a model artifact",
    )
    train_artifact.add_argument("--input-file", type=Path, default=None)
    train_artifact.add_argument("--input-dir", type=Path, default=DEFAULT_FEATURED_DIR)
    _add_dataset_version_arg(train_artifact)
    _add_model_config_arg(train_artifact)
    _add_training_args(train_artifact)
    _add_artifact_args(train_artifact)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "full":
        artifact = _run_full(args)
        print(json.dumps(_artifact_payload(artifact), indent=2))
    elif args.command == "scrape":
        csv_path, manifest_path = _run_scrape(args)
        print(json.dumps({"csv_path": str(csv_path), "manifest_path": str(manifest_path)}, indent=2))
    elif args.command == "features":
        results = _run_features(args)
        print(json.dumps([_feature_result_payload(result) for result in results], indent=2))
    elif args.command == "train-artifact":
        artifact = _run_train_artifact(args)
        print(json.dumps(_artifact_payload(artifact), indent=2))
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unknown command: {args.command}")


def _run_full(args: argparse.Namespace) -> ModelArtifact:
    scraped_csv_path, _ = _run_scrape(args)
    feature_results = _run_features(
        argparse.Namespace(
            input_file=scraped_csv_path,
            input_dir=DEFAULT_SCRAPED_DIR,
            output_dir=DEFAULT_FEATURED_DIR,
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
    )
    if len(feature_results) != 1:
        raise RuntimeError("Expected full workflow to produce exactly one featured dataset.")

    return _train_and_save(
        args=args,
        dataset_paths=[feature_results[0].output_path],
    )


def _run_scrape(args: argparse.Namespace) -> tuple[Path, Path]:
    scraper = RawResultScraper(
        output_root=getattr(args, "output_root", None),
        dataset_dir=getattr(args, "scraped_dir", DEFAULT_SCRAPE_OUTPUT_DIR),
    )
    return scraper.export(
        experiment_name=args.experiment_name,
        dataset_version=args.dataset_version,
    )


def _run_features(args: argparse.Namespace) -> list[Any]:
    source_paths = _resolve_feature_source_paths(args.input_file, args.input_dir)
    client = OllamaEmbeddingClient(
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
    )
    builder = FeatureBuilder(
        embedding_client=client,
        output_dir=args.output_dir,
    )
    return [builder.build_file(path) for path in source_paths]


def _run_train_artifact(args: argparse.Namespace) -> ModelArtifact:
    dataset_paths = _resolve_dataset_paths(
        input_file=args.input_file,
        input_dir=args.input_dir,
        dataset_version=args.dataset_version,
    )
    return _train_and_save(args=args, dataset_paths=dataset_paths)


def _train_and_save(
    *,
    args: argparse.Namespace,
    dataset_paths: list[Path],
) -> ModelArtifact:
    model_config = _load_model_config(args.model_config)
    training_config = _training_config_from_args(args, model_config)
    trainer = VotingModelTrainer(config=training_config)
    training_result = trainer.train(dataset_paths)
    manager = ModelArtifactManager(_artifact_config_from_args(args, model_config))
    return manager.save_training_result(
        training_result=training_result,
        training_config=training_config,
        model_version=args.model_version,
    )


def _training_config_from_args(
    args: argparse.Namespace,
    model_config: ModelConfig,
) -> TrainingConfig:
    return TrainingConfig(
        feature_columns=[
            "candidate_turn_position",
            "embedding_similarity_to_others_mean",
            "embedding_similarity_to_others_std",
            "embedding_similarity_rank_low_to_high",
            "embedding_similarity_to_previous_mean",
        ],
        model=model_config,
        test_size=args.test_size,
        random_state=args.random_state,
    )


def _artifact_config_from_args(
    args: argparse.Namespace,
    model_config: ModelConfig,
) -> ArtifactConfig:
    return ArtifactConfig(
        model_name=model_config.name,
        artifact_root=args.artifact_root,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name,
        enable_mlflow=not args.disable_mlflow,
    )


def _resolve_feature_source_paths(
    input_file: Path | None,
    input_dir: Path,
) -> list[Path]:
    if input_file is not None:
        if not input_file.exists():
            raise FileNotFoundError(f"Scraped dataset not found: {input_file}")
        return [input_file]

    paths = sorted(input_dir.glob("voting_candidates_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No scraped datasets found at {input_dir}.")
    return paths


def _artifact_payload(artifact: ModelArtifact) -> dict[str, Any]:
    return {
        "model_version": artifact.model_version,
        "model_dir": str(artifact.model_dir),
        "model_path": str(artifact.model_path),
        "metadata_path": str(artifact.metadata_path),
        "mlflow_run_id": artifact.mlflow_run_id,
    }


def _feature_result_payload(result: Any) -> dict[str, Any]:
    return {
        "source_path": str(result.source_path),
        "output_path": str(result.output_path),
        "manifest_path": str(result.manifest_path),
        "row_count": result.row_count,
    }


def _add_dataset_version_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-version", default=None)


def _add_embedding_args(
    parser: argparse.ArgumentParser,
    embedding_config: dict[str, str],
) -> None:
    parser.add_argument(
        "--model",
        default=embedding_config.get("embedding_model_name", "nomic-embed-text"),
    )
    parser.add_argument(
        "--base-url",
        default=embedding_config.get("embedding_model_server_url", "http://localhost:11434"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)


def _add_model_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_DIR / "random_forest.json",
        help="Path to the JSON model configuration to use for training.",
    )


def _add_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_MLFLOW_EXPERIMENT_NAME)
    parser.add_argument("--disable-mlflow", action="store_true")

def _load_model_config(path: Path) -> ModelConfig:
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model config: {path}") from exc

    required_fields = {"name", "estimator", "parameters"}
    missing_fields = required_fields - payload.keys()
    if missing_fields:
        raise ValueError(f"Missing required fields in model config: {missing_fields}")
    if not isinstance(payload["parameters"], dict):
        raise ValueError(f"'parameters' field must be a dictionary in model config: {path}")

    return ModelConfig(
        name=str(payload["name"]),
        estimator=str(payload["estimator"]),
        parameters=payload["parameters"]
    )

if __name__ == "__main__":
    main()
