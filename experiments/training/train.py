from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from .evaluate import VotingModelEvaluator
from .features import FEATURE_COLUMNS


TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = TRAINING_DIR / "datasets" / "featured"
INPUT_PATTERN = "voting_features_*.csv"
INPUT_PREFIX = "voting_features_"

LABEL_COLUMN = "is_imposter"
METADATA_COLUMNS = [
    "experiment_name",
    "case_id",
    "round_id",
    "turn_id",
    "secret_word",
    "imposter_was",
    "candidate_agent_id",
    "candidate_agent_name",
    "candidate_clue",
    "all_clues_json",
]

@dataclass(frozen=True)
class ModelConfig:
    name: str
    estimator: str
    parameters: dict[str, Any]

@dataclass(frozen=True)
class TrainingConfig:
    feature_columns: list[str]
    model: ModelConfig
    label_column: str = LABEL_COLUMN
    test_size: float = 0.2
    random_state: int = 42

@dataclass(frozen=True)
class TrainingResult:
    model: Pipeline
    metrics: dict[str, Any]
    feature_columns: list[str]
    dataset_paths: list[Path]
    train_row_count: int
    test_row_count: int
    train_round_count: int
    test_round_count: int


class VotingModelTrainer:
    def __init__(
        self,
        config: TrainingConfig | None = None,
        evaluator: VotingModelEvaluator | None = None,
    ) -> None:
        self.config = config or TrainingConfig(feature_columns=FEATURE_COLUMNS)
        self.evaluator = evaluator or VotingModelEvaluator()

    def train(self, dataset_paths: list[Path]) -> TrainingResult:
        rows = self.load_featured_datasets(dataset_paths)
        labeled_rows = self.prepare_training_rows(rows)
        train_rows, test_rows = self.split_by_round(labeled_rows)

        X_train, y_train, _ = self.frame_parts(train_rows)
        X_test, y_test, test_metadata = self.frame_parts(test_rows)

        model = self.build_pipeline()
        model.fit(X_train, y_train)

        evaluation = self.evaluator.evaluate(
            model=model,
            X=X_test,
            y=y_test,
            metadata=test_metadata,
            feature_columns=self.config.feature_columns,
        )
        metrics = {
            **evaluation.metrics,
            "train_row_count": len(train_rows),
            "test_row_count": len(test_rows),
            "train_round_count": _round_count(train_rows),
            "test_round_count": _round_count(test_rows),
            "positive_rate_train": _positive_rate(train_rows, self.config.label_column),
            "positive_rate_test": _positive_rate(test_rows, self.config.label_column),
        }

        return TrainingResult(
            model=model,
            metrics=metrics,
            feature_columns=self.config.feature_columns,
            dataset_paths=dataset_paths,
            train_row_count=len(train_rows),
            test_row_count=len(test_rows),
            train_round_count=_round_count(train_rows),
            test_round_count=_round_count(test_rows),
        )

    def load_featured_datasets(self, dataset_paths: list[Path]) -> list[dict[str, str]]:
        if not dataset_paths:
            raise ValueError("At least one featured dataset path is required.")

        rows: list[dict[str, str]] = []
        for path in dataset_paths:
            if not path.exists():
                raise FileNotFoundError(f"Featured dataset not found: {path}")
            with path.open(encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))

        return rows

    def prepare_training_rows(
        self,
        rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if not rows:
            raise ValueError("No rows found in featured dataset.")

        missing_features = [
            column
            for column in self.config.feature_columns
            if column not in rows[0]
        ]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")

        if self.config.label_column not in rows[0]:
            raise ValueError(f"Missing label column: {self.config.label_column}")

        labeled_rows = [
            row
            for row in rows
            if row.get(self.config.label_column, "").strip() != ""
        ]
        if not labeled_rows:
            raise ValueError("No labeled rows found. is_imposter is blank for all rows.")

        labels = {_label(row, self.config.label_column) for row in labeled_rows}
        if labels != {0, 1}:
            raise ValueError(
                "Training requires both classes in labeled rows. "
                f"Found labels: {sorted(labels)}"
            )

        if _round_count(labeled_rows) < 2:
            raise ValueError("Training requires at least two unique round_id values.")

        for row in labeled_rows:
            for column in self.config.feature_columns:
                _float_value(row, column)

        return labeled_rows

    def split_by_round(
        self,
        rows: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        groups = [row.get("round_id", "") for row in rows]
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        train_indices, test_indices = next(splitter.split(rows, groups=groups))
        train_rows = [rows[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]

        if len({_label(row, self.config.label_column) for row in test_rows}) < 2:
            raise ValueError(
                "The test split contains fewer than two classes. "
                "Use more rounds or a different random state."
            )

        return train_rows, test_rows

    def frame_parts(
        self,
        rows: list[dict[str, str]],
    ) -> tuple[list[list[float]], list[int], list[dict[str, str]]]:
        X = [
            [_float_value(row, column) for column in self.config.feature_columns]
            for row in rows
        ]
        y = [_label(row, self.config.label_column) for row in rows]
        metadata = [
            {
                column: row.get(column, "")
                for column in METADATA_COLUMNS
                if column in row
            }
            for row in rows
        ]
        return X, y, metadata

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                (
                    "model",
                    build_estimator(self.config.model)
                )
            ]
        )
    
def build_estimator(config: ModelConfig) -> ClassifierMixin:
    if config.estimator == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**config.parameters)
    elif config.estimator == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**config.parameters)
    elif config.estimator == "svm":
        from sklearn.svm import SVC
        return SVC(**config.parameters)
    
    raise ValueError(f"Unsupported estimator: {config.estimator}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first random forest voting model from featured CSVs."
    )
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--class-weight", default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = _resolve_dataset_paths(
        input_file=args.input_file,
        input_dir=args.input_dir,
        dataset_version=args.dataset_version,
    )
    config = TrainingConfig(
        feature_columns=FEATURE_COLUMNS,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=args.class_weight if args.class_weight else None,
    )
    trainer = VotingModelTrainer(config=config)
    result = trainer.train(dataset_paths)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


def _resolve_dataset_paths(
    *,
    input_file: Path | None,
    input_dir: Path,
    dataset_version: str | None,
) -> list[Path]:
    if input_file is not None:
        return [input_file]

    if dataset_version is not None:
        return [input_dir / f"{INPUT_PREFIX}{dataset_version}.csv"]

    paths = sorted(input_dir.glob(INPUT_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No featured datasets found at {input_dir / INPUT_PATTERN}.")
    return paths


def _label(row: dict[str, str], label_column: str) -> int:
    value = row.get(label_column, "").strip()
    if value not in {"0", "1"}:
        raise ValueError(f"Expected {label_column} to be 0 or 1, got {value!r}.")
    return int(value)


def _float_value(row: dict[str, str], column: str) -> float:
    value = row.get(column, "")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected numeric value for {column}, got {value!r}.") from exc


def _round_count(rows: list[dict[str, str]]) -> int:
    return len({row.get("round_id", "") for row in rows})


def _positive_rate(rows: list[dict[str, str]], label_column: str) -> float:
    if not rows:
        return 0.0
    return sum(_label(row, label_column) for row in rows) / len(rows)


if __name__ == "__main__":
    main()
