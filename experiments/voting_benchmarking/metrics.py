from __future__ import annotations

from collections import Counter
from statistics import mean

from schemas import (
    BenchmarkSummary,
    RoundArtifactRecord,
    SemanticFeatureRecord,
    VoteRecord,
)


def summarize_results(
    technique: str,
    rounds: list[RoundArtifactRecord],
    votes: list[VoteRecord],
    semantic_features: list[SemanticFeatureRecord],
) -> BenchmarkSummary:
    completed_rounds = [round_record for round_record in rounds if round_record.success]
    failed_rounds = [round_record for round_record in rounds if not round_record.success]
    random_chance = _average(
        [1.0 / round_record.num_players for round_record in completed_rounds if round_record.num_players]
    )
    agent_vote_detection = _agent_vote_detection_rate(votes)
    agent_only_detection = _agent_only_group_detection_rate(votes)
    return BenchmarkSummary(
        technique=technique,
        completed_rounds=len(completed_rounds),
        failed_rounds=len(failed_rounds),
        agent_imposter_rounds=sum(
            1 for round_record in completed_rounds if round_record.imposter_kind == "agent"
        ),
        human_imposter_rounds=sum(
            1 for round_record in completed_rounds if round_record.imposter_kind == "human"
        ),
        average_latency_ms=_average([round_record.latency_ms for round_record in rounds]),
        round_success_rate=_rate([round_record.success for round_record in rounds]),
        agent_vote_detection_rate=agent_vote_detection,
        agent_only_group_detection_rate=agent_only_detection,
        group_detection_rate=_rate(
            [
                vote.group_voted_is_imposter is True
                for vote in votes
                if vote.group_voted_is_imposter is not None
            ]
        ),
        random_chance_detection_rate=random_chance,
        detection_lift_over_random=agent_only_detection - random_chance,
        mean_hint_to_secret_similarity=_semantic_average(
            semantic_features,
            "hint_to_secret_similarity",
        ),
        mean_non_imposter_clue_to_secret_similarity=_semantic_average(
            [
                feature
                for feature in semantic_features
                if feature.role == "non_imposter"
            ],
            "clue_to_secret_similarity",
        ),
        mean_imposter_clue_to_secret_similarity=_semantic_average(
            [feature for feature in semantic_features if feature.role == "imposter"],
            "clue_to_secret_similarity",
        ),
        mean_non_imposter_pairwise_similarity=_semantic_average(
            semantic_features,
            "non_imposter_pairwise_similarity",
        ),
        mean_imposter_outlier_score=_semantic_average(
            semantic_features,
            "imposter_outlier_score",
        ),
        mean_separability_margin=_semantic_average(
            semantic_features,
            "separability_margin",
        ),
        detection_rate_by_imposter_position=_detection_rate_by_imposter_position(
            completed_rounds,
            votes,
        ),
    )


def _agent_vote_detection_rate(votes: list[VoteRecord]) -> float:
    flags: list[bool] = []
    for vote in votes:
        flags.extend(
            agent_vote.voted_for_is_imposter is True
            for agent_vote in vote.agent_votes
            if agent_vote.voted_for_is_imposter is not None
        )
    return _rate(flags)


def _agent_only_group_detection_rate(votes: list[VoteRecord]) -> float:
    flags: list[bool] = []
    for vote in votes:
        counts = Counter(
            agent_vote.voted_for_is_imposter
            for agent_vote in vote.agent_votes
            if agent_vote.voted_for_is_imposter is not None
        )
        if not counts:
            continue

        highest_total = max(counts.values())
        leaders = [
            voted_for_is_imposter
            for voted_for_is_imposter, vote_total in counts.items()
            if vote_total == highest_total
        ]
        if len(leaders) == 1:
            flags.append(leaders[0] is True)
    return _rate(flags)


def _detection_rate_by_imposter_position(
    rounds: list[RoundArtifactRecord],
    votes: list[VoteRecord],
) -> dict[str, float]:
    votes_by_round_id = {vote.round_id: vote for vote in votes}
    flags_by_position: dict[str, list[bool]] = {}

    for round_record in rounds:
        if round_record.round_id is None or round_record.imposter_player_id is None:
            continue

        vote = votes_by_round_id.get(round_record.round_id)
        if vote is None:
            continue

        position = next(
            (
                player.position
                for player in round_record.playing_order
                if player.player_id == round_record.imposter_player_id
            ),
            None,
        )
        if position is None:
            continue

        flags_by_position.setdefault(str(position), []).append(
            vote.group_voted_is_imposter is True
        )

    return {
        position: _rate(flags)
        for position, flags in sorted(flags_by_position.items(), key=lambda item: int(item[0]))
    }


def _semantic_average(
    features: list[SemanticFeatureRecord],
    field_name: str,
) -> float | None:
    values = [
        value
        for feature in features
        if (value := getattr(feature, field_name)) is not None
    ]
    if not values:
        return None
    return float(mean(values))


def _average(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for flag in flags if flag) / len(flags))
