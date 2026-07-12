from __future__ import annotations

import math
from typing import Any
from app.services.voting.schemas import VotingFeatureInput

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError as exc:  # pragma: no cover - exercised only without sklearn.
    raise ImportError(
        "features.py requires scikit-learn. Install scikit-learn before running "
        "feature generation."
    ) from exc

FEATURE_SCHEMA_VERSION = 1
FEATURE_BUILDER_NAME = "embedding_similarity_v1"
FEATURE_COLUMNS = [
    "candidate_turn_position",
    "embedding_similarity_to_others_mean",
    "embedding_similarity_to_others_std",
    "embedding_similarity_rank_low_to_high",
    "embedding_similarity_to_previous_mean",
]


class VotingFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(
        self,
        X: list[VotingFeatureInput],
        y: Any = None,
    ) -> VotingFeatureTransformer:
        return self

    def transform(self, X: list[VotingFeatureInput]) -> list[list[float]]:
        grouped = _group_indices_by_round(X)
        similarity_means_by_index: dict[int, float] = {}

        for indices in grouped.values():
            for index in indices:
                similarities = _candidate_to_other_similarities(X[index])
                similarity_means_by_index[index] = _mean(similarities)

        ranks_by_index = _rank_similarity_means(grouped, similarity_means_by_index)

        feature_matrix: list[list[float]] = []
        for index, record in enumerate(X):
            similarities = _candidate_to_other_similarities(record)
            previous_similarities = _candidate_to_previous_similarities(record)
            feature_matrix.append(
                [
                    float(record["candidate_turn_position"]),
                    _mean(similarities),
                    _std(similarities),
                    float(ranks_by_index[index]),
                    _mean(previous_similarities),
                ]
            )

        return feature_matrix



def _candidate_to_other_similarities(record: VotingFeatureInput) -> list[float]:
    return [
        _cosine_similarity(record["candidate_embedding"], embedding)
        for embedding in record["other_embeddings"]
    ]


def _candidate_to_previous_similarities(record: VotingFeatureInput) -> list[float]:
    return [
        _cosine_similarity(record["candidate_embedding"], embedding)
        for embedding in record["previous_embeddings"]
    ]


def _group_indices_by_round(
    records: list[VotingFeatureInput],
) -> dict[tuple[str, str], list[int]]:
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
