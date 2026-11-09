import asyncio

from app.services import latency
from collections import Counter
from typing import Any
from app.services.agents.inference import (
    AgentConfig, InferenceClient, InferenceRequest, InferenceServiceError,
)
from app.services.agents.discussion import DiscussionVote, discussion_agent, player_context
from app.services.voting.schemas import AgentVoteResponse, VoteCountResponse, VoteResponse

async def submit_round_vote(
    round_state: Any,
    agents: list[AgentConfig],
    voted_agent: AgentConfig | None,
    settings: Any,
    *, http_client=None,
) -> VoteResponse:
    agent_votes = await _build_agent_votes(
        round_state,
        agents,
        settings,
        http_client=http_client,
    )
    vote_counts, group_voted_player_id = _tally_round_votes(
        voted_agent.id if voted_agent is not None else None,
        agent_votes,
        round_state.player_names_by_id,
    )

    player_names_by_id = round_state.player_names_by_id
    imposter_was = player_names_by_id[round_state.imposter_player_id]
    group_voted_player_name = (
        player_names_by_id.get(group_voted_player_id)
        if group_voted_player_id is not None
        else None
    )
    imposter_won = group_voted_player_id != round_state.imposter_player_id

    return VoteResponse(
        voted_agent_id=voted_agent.id if voted_agent is not None else None,
        voted_agent_name=player_names_by_id[voted_agent.id] if voted_agent is not None else None,
        secret_word=round_state.secret_word,
        imposter_was=imposter_was,
        agent_votes=agent_votes,
        vote_counts=vote_counts,
        group_voted_player_id=group_voted_player_id,
        group_voted_player_name=group_voted_player_name,
        imposter_won=imposter_won,
        round_winner="imposter" if imposter_won else "players",
    )


def _tally_round_votes(
    human_voted_agent_id: str | None,
    agent_votes: list[AgentVoteResponse],
    player_names_by_id: dict[str, str],
) -> tuple[list[VoteCountResponse], str | None]:
    vote_counter: Counter[str] = Counter()
    if human_voted_agent_id is not None:
        vote_counter[human_voted_agent_id] += 1

    for agent_vote in agent_votes:
        target_id = agent_vote.voted_for if agent_vote.voted_for in player_names_by_id else None
        if target_id is not None:
            vote_counter[target_id] += 1

    if not vote_counter:
        return [], None

    highest_vote_total = max(vote_counter.values())
    leading_player_ids = [
        player_id
        for player_id, vote_total in vote_counter.items()
        if vote_total == highest_vote_total
    ]
    group_voted_player_id = leading_player_ids[0] if len(leading_player_ids) == 1 else None

    vote_counts = [
        VoteCountResponse(
            player_id=player_id,
            player_name=player_names_by_id.get(player_id, player_id),
            votes=vote_total,
        )
        for player_id, vote_total in sorted(
            vote_counter.items(),
            key=lambda item: (-item[1], player_names_by_id.get(item[0], item[0])),
        )
    ]

    return vote_counts, group_voted_player_id


async def _build_agent_votes(state, agents, settings, *, http_client=None):
    client = InferenceClient(settings, http_client=http_client)
    async def vote(agent):
        with latency.span("agent_vote", round_id=state.id, phase="voting",
                          agent_id=agent.id, turn_id=f"vote:{agent.id}"):
            eligible = set(state.playing_order) - {agent.id}

            def validate(result):
                if result.target_id not in eligible:
                    raise ValueError("Vote must target another participant")

            decision, result = await client.generate_structured(InferenceRequest(
                agent=discussion_agent(agent),
                prompt=("Discussion has ended. Vote for another participant using the clues and "
                        "public discussion. If you are the imposter, try to avoid being eliminated. "
                        "Return only target_id. GAME_CONTEXT:\n" + player_context(state, agent.id)),
            ), DiscussionVote, validate=validate)
            return AgentVoteResponse(
                voter_agent_id=agent.id, voter_agent_name=state.player_names_by_id[agent.id],
                voted_for=decision.target_id,
                inference_mode=result.inference_mode,
            )

    # Await every task before closing the shared HTTP client, including on failure.
    results = await asyncio.gather(*(vote(agent) for agent in agents), return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            raise InferenceServiceError("Discussion-aware voting failed; please retry") from result
    return results
