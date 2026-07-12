from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import joblib

from .train import TrainingConfig, ModelConfig, TrainingResult, VotingModelTrainer, _resolve_dataset_paths


TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = TRAINING_DIR / "trained_models"
DEFAULT_MLFLOW_TRACKING_URI = str(TRAINING_DIR / "mlruns")
DEFAULT_MLFLOW_EXPERIMENT_NAME = "voting_model"


@dataclass(frozen=True)
class ArtifactConfig:
    model_name: str
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI
    mlflow_experiment_name: str = DEFAULT_MLFLOW_EXPERIMENT_NAME
    enable_mlflow: bool = True


@dataclass(frozen=True)
class ModelArtifact:
    model_version: str
    model_dir: Path
    model_path: Path
    metadata_path: Path
    mlflow_run_id: str | None


class ModelArtifactManager:
    def __init__(self, config: ArtifactConfig | None = None) -> None:
        self.config = config or ArtifactConfig()

    def save_training_result(
        self,
        *,
        training_result: TrainingResult,
        training_config: TrainingConfig,
        model_version: str | None = None,
    ) -> ModelArtifact:
        version = model_version or _default_model_version(self.config.model_name)
        model_dir = self.config.artifact_root / version
        model_dir.mkdir(parents=True, exist_ok=False)

        model_path = model_dir / "model.joblib"
        metadata_path = model_dir / "metadata.json"

        joblib.dump(training_result.model, model_path)
        metadata = self._metadata(
            training_result=training_result,
            training_config=training_config,
            model_version=version,
            model_path=model_path,
        )
        _write_json(metadata_path, metadata)

        mlflow_run_id = None
        if self.config.enable_mlflow:
            mlflow_run_id = self._log_to_mlflow(
                training_result=training_result,
                training_config=training_config,
                model_version=version,
                model_path=model_path,
                metadata_path=metadata_path,
            )

        return ModelArtifact(
            model_version=version,
            model_dir=model_dir,
            model_path=model_path,
            metadata_path=metadata_path,
            mlflow_run_id=mlflow_run_id,
        )

    def _metadata(
        self,
        *,
        training_result: TrainingResult,
        training_config: TrainingConfig,
        model_version: str,
        model_path: Path,
    ) -> dict[str, Any]:
        return {
            "model_version": model_version,
            "model_name": self.config.model_name,
            "created_at": datetime.now(UTC).isoformat(),
            "model_path": str(model_path),
            "feature_columns": training_result.feature_columns,
            "dataset_paths": [str(path) for path in training_result.dataset_paths],
            "training_config": asdict(training_config),
            "metrics": training_result.metrics,
            "train_row_count": training_result.train_row_count,
            "test_row_count": training_result.test_row_count,
            "train_round_count": training_result.train_round_count,
            "test_round_count": training_result.test_round_count,
        }

    def _log_to_mlflow(
        self,
        *,
        training_result: TrainingResult,
        training_config: TrainingConfig,
        model_version: str,
        model_path: Path,
        metadata_path: Path,
    ) -> str:
        try:
            import mlflow
            import mlflow.sklearn
        except ImportError as exc:
            raise ImportError(
                "MLflow logging is enabled, but mlflow is not installed. "
                "Install mlflow or pass --disable-mlflow."
            ) from exc

        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment_name)

        with mlflow.start_run(run_name=model_version) as run:
            mlflow.log_params(_flatten_params("training", asdict(training_config)))
            mlflow.log_param("model_name", self.config.model_name)
            mlflow.log_param("model_version", model_version)
            mlflow.log_param(
                "dataset_paths",
                json.dumps([str(path) for path in training_result.dataset_paths]),
            )

            for name, value in _flatten_numeric_metrics(training_result.metrics).items():
                mlflow.log_metric(name, value)

            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(metadata_path))
            mlflow.sklearn.log_model(
                sk_model=training_result.model,
                artifact_path="model",
            )
            return run.info.run_id

def _load_model_config(path: Path) -> ModelConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Model config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config_data = json.load(f)
    return ModelConfig(**config_data)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a voting model and save a versioned artifact."
    )
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=TRAINING_DIR / "datasets" / "featured")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--mlflow-experiment-name", default=DEFAULT_MLFLOW_EXPERIMENT_NAME)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = _resolve_dataset_paths(
        input_file=args.input_file,
        input_dir=args.input_dir,
        dataset_version=args.dataset_version,
    )
    training_config = TrainingConfig(
        feature_columns=[
            "candidate_turn_position",
            "embedding_similarity_to_others_mean",
            "embedding_similarity_to_others_std",
            "embedding_similarity_rank_low_to_high",
            "embedding_similarity_to_previous_mean",
        ],
        model=_load_model_config(args.model_config),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    trainer = VotingModelTrainer(config=training_config)
    training_result = trainer.train(dataset_paths)
    manager = ModelArtifactManager(
        ArtifactConfig(
            model_name=args.model_config["name"],
            artifact_root=args.artifact_root,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            enable_mlflow=not args.disable_mlflow,
        )
    )
    artifact = manager.save_training_result(
        training_result=training_result,
        training_config=training_config,
        model_version=args.model_version,
    )
    print(json.dumps(_artifact_summary(artifact, training_result.metrics), indent=2))


def _artifact_summary(
    artifact: ModelArtifact,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_version": artifact.model_version,
        "model_dir": str(artifact.model_dir),
        "model_path": str(artifact.model_path),
        "metadata_path": str(artifact.metadata_path),
        "mlflow_run_id": artifact.mlflow_run_id,
        "metrics": metrics,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _default_model_version(model_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in model_name.strip()
    ).strip("_")
    return f"{safe_name or 'model'}_{timestamp}"


def _flatten_params(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_params(name, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[name] = "null" if value is None else value
        else:
            flattened[name] = json.dumps(value, default=str)
    return flattened


def _flatten_numeric_metrics(
    metrics: dict[str, Any],
    prefix: str | None = None,
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_numeric_metrics(value, name))
        elif isinstance(value, bool):
            flattened[name] = float(value)
        elif isinstance(value, (int, float)):
            flattened[name] = float(value)
    return flattened


if __name__ == "__main__":
    main()
