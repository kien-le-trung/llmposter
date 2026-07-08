from schemas import (
    AgentVoteRecord,
    PlayerRecord,
    RoundArtifactRecord,
    VoteRecord,
)
from metrics import summarize_results


def make_round(
    round_id: str,
    imposter_player_id: str = "agent_b",
    imposter_kind: str = "agent",
) -> RoundArtifactRecord:
    return RoundArtifactRecord(
        run_id="run-1",
        technique="few_shot",
        round_id=round_id,
        secret_word="music",
        imposter_hint="sound",
        human_clue="rhythm notes",
        status="ready_to_vote",
        success=True,
        latency_ms=100.0,
        playing_order=[
            PlayerRecord(
                player_id="human",
                player_name="You",
                player_kind="human",
                position=0,
            ),
            PlayerRecord(
                player_id="agent_a",
                player_name="Agent A",
                player_kind="agent",
                position=1,
            ),
            PlayerRecord(
                player_id="agent_b",
                player_name="Agent B",
                player_kind="agent",
                position=2,
            ),
        ],
        imposter_player_id=imposter_player_id,
        imposter_player_name="Agent B",
        imposter_kind=imposter_kind,
        num_players=3,
    )


def make_vote(
    round_id: str,
    first_vote_hits: bool,
    second_vote_hits: bool,
    group_hits: bool,
) -> VoteRecord:
    return VoteRecord(
        run_id="run-1",
        technique="few_shot",
        round_id=round_id,
        voting_algorithm="embedding_distance_v1",
        human_vote_strategy="first_agent_placeholder",
        human_voted_player_id="agent_a",
        group_voted_player_id="agent_b" if group_hits else "agent_a",
        group_voted_player_name="Agent B" if group_hits else "Agent A",
        group_voted_is_imposter=group_hits,
        imposter_won=not group_hits,
        round_winner="players" if group_hits else "imposter",
        agent_votes=[
            AgentVoteRecord(
                voter_agent_id="agent_a",
                voter_agent_name="Agent A",
                voted_for_player_id="agent_b" if first_vote_hits else "human",
                voted_for_player_name="Agent B" if first_vote_hits else "You",
                voted_for_is_imposter=first_vote_hits,
                inference_mode="embedding",
            ),
            AgentVoteRecord(
                voter_agent_id="agent_b",
                voter_agent_name="Agent B",
                voted_for_player_id="agent_b" if second_vote_hits else "agent_a",
                voted_for_player_name="Agent B" if second_vote_hits else "Agent A",
                voted_for_is_imposter=second_vote_hits,
                inference_mode="embedding",
            ),
        ],
        vote_counts=[],
    )


def test_detection_rates() -> None:
    summary = summarize_results(
        "few_shot",
        [make_round("round-1"), make_round("round-2")],
        [
            make_vote("round-1", True, False, True),
            make_vote("round-2", False, False, False),
        ],
        [],
    )

    assert summary.completed_rounds == 2
    assert summary.agent_vote_detection_rate == 0.25
    assert summary.agent_only_group_detection_rate == 0.0
    assert summary.group_detection_rate == 0.5
    assert summary.random_chance_detection_rate == 1 / 3


def test_failed_rounds_and_empty_inputs() -> None:
    failed_round = make_round("round-1")
    failed_round.success = False
    failed_round.status = "error"

    summary = summarize_results("few_shot", [failed_round], [], [])

    assert summary.completed_rounds == 0
    assert summary.failed_rounds == 1
    assert summary.round_success_rate == 0.0
    assert summary.agent_vote_detection_rate == 0.0


def test_detection_rate_by_imposter_position() -> None:
    summary = summarize_results(
        "few_shot",
        [make_round("round-1"), make_round("round-2")],
        [
            make_vote("round-1", True, True, True),
            make_vote("round-2", False, False, False),
        ],
        [],
    )

    assert summary.detection_rate_by_imposter_position == {"2": 0.5}
