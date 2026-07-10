from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:  # pragma: no cover - exercised only without sklearn.
    raise ImportError(
        "evaluate.py requires scikit-learn. Install scikit-learn before running "
        "model evaluation."
    ) from exc


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: list[int]
    probabilities: list[float]


class VotingModelEvaluator:
    def evaluate(
        self,
        *,
        model: Any,
        X: list[list[float]],
        y: list[int],
        metadata: list[dict[str, str]],
        feature_columns: list[str],
    ) -> EvaluationResult:
        if not X:
            raise ValueError("Cannot evaluate an empty dataset.")
        if len(X) != len(y) or len(X) != len(metadata):
            raise ValueError("X, y, and metadata must have the same length.")

        predictions = [int(value) for value in model.predict(X)]
        probabilities = _positive_probabilities(model, X)
        metrics = self._metrics(
            y_true=y,
            y_pred=predictions,
            y_probability=probabilities,
            metadata=metadata,
            feature_importances=_feature_importances(model, feature_columns),
        )
        return EvaluationResult(
            metrics=metrics,
            predictions=predictions,
            probabilities=probabilities,
        )

    def _metrics(
        self,
        *,
        y_true: list[int],
        y_pred: list[int],
        y_probability: list[float],
        metadata: list[dict[str, str]],
        feature_importances: dict[str, float],
    ) -> dict[str, Any]:
        labels = [0, 1]
        metrics: dict[str, Any] = {
            "row_count": len(y_true),
            "positive_rate": _mean([float(value) for value in y_true]),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
            ).tolist(),
            "round_top1_accuracy": _round_top1_accuracy(
                y_true=y_true,
                y_probability=y_probability,
                metadata=metadata,
            ),
            "feature_importances": feature_importances,
        }

        if len(set(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_probability))
        else:
            metrics["roc_auc"] = None

        return metrics


def _positive_probabilities(model: Any, X: list[list[float]]) -> list[float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = [int(value) for value in model.classes_]
        if 1 not in classes:
            return [0.0 for _ in X]
        positive_index = classes.index(1)
        return [float(row[positive_index]) for row in probabilities]

    return [float(value) for value in model.predict(X)]


def _feature_importances(model: Any, feature_columns: list[str]) -> dict[str, float]:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model", model)

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return {}

    return {
        feature: float(importance)
        for feature, importance in zip(feature_columns, importances, strict=True)
    }


def _round_top1_accuracy(
    *,
    y_true: list[int],
    y_probability: list[float],
    metadata: list[dict[str, str]],
) -> float:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(metadata):
        key = (row.get("round_id", ""), row.get("turn_id", ""))
        grouped.setdefault(key, []).append(index)

    if not grouped:
        return 0.0

    correct = 0
    for indices in grouped.values():
        best_index = max(indices, key=lambda index: (y_probability[index], -index))
        if y_true[best_index] == 1:
            correct += 1

    return correct / len(grouped)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
