from collections import Counter
from typing import Any
from app.services.agents.inference import AgentConfig, InferenceClient, InferenceServiceError
from app.services.voting.schemas import VotingFeatureInput, AgentVoteResponse, VoteCountResponse, VoteResponse
from app.services.voting.predictor import VotingModelPredictor, get_voting_predictor
from app.services.voting.features import VotingFeatureTransformer

HUMAN_PLAYER_ID = "human"
HUMAN_PLAYER_NAME = "You"

ROUND_EMBEDDINGS: dict[str, dict[str, list[float]]] = {}
EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}

class VotingStateError(Exception):
    """Raised when the current round state cannot produce votes."""


async def submit_round_vote(
    round_state: Any,
    agents: list[AgentConfig],
    voted_agent: AgentConfig | None,
    human_clue: str | None,
    settings: Any,
) -> VoteResponse:
    agent_votes = await _build_agent_votes(
        round_state,
        agents,
        settings,
        human_clue,
    )
    vote_counts, group_voted_player_id = _tally_round_votes(
        voted_agent.id if voted_agent is not None else None,
        agent_votes,
        agents,
    )

    round_state.voted_agent_id = voted_agent.id if voted_agent is not None else None
    round_state.status = "complete"

    imposter_agent = next(
        (agent for agent in agents if agent.id == round_state.imposter_player_id),
        None,
    )
    imposter_was = imposter_agent.name if imposter_agent is not None else HUMAN_PLAYER_NAME
    player_names_by_id = _get_player_names_by_id(agents)
    group_voted_player_name = (
        player_names_by_id.get(group_voted_player_id)
        if group_voted_player_id is not None
        else None
    )
    imposter_won = group_voted_player_id != round_state.imposter_player_id

    return VoteResponse(
        voted_agent_id=voted_agent.id if voted_agent is not None else None,
        voted_agent_name=voted_agent.name if voted_agent is not None else None,
        secret_word=round_state.secret_word,
        imposter_was=imposter_was,
        agent_votes=agent_votes,
        vote_counts=vote_counts,
        group_voted_player_id=group_voted_player_id,
        group_voted_player_name=group_voted_player_name,
        imposter_won=imposter_won,
        round_winner="imposter" if imposter_won else "players",
    )


async def _build_agent_votes(
    round_state: Any,
    agents: list[AgentConfig],
    settings: Any,
    human_clue: str | None = None,
) -> list[AgentVoteResponse]:
    if not round_state.turns:
        return []

    opening_turn = round_state.turns[0]
    agent_names_by_id = {agent.id: agent.name for agent in agents}
    phrase_by_player_id = {
        response.agent_id: response.agent_response for response in opening_turn.responses
    }

    if HUMAN_PLAYER_ID in round_state.playing_order:
        resolved_human_clue = human_clue or round_state.human_clue
        if resolved_human_clue is None:
            raise VotingStateError("Human clue has not been submitted")

        phrase_by_player_id[HUMAN_PLAYER_ID] = resolved_human_clue

    embeddings_by_player_id = await _get_or_create_round_embeddings(
        round_state.id,
        phrase_by_player_id,
        settings,
    )
    predictor = get_voting_predictor(settings.ml_voting_model_path)

    votes: list[AgentVoteResponse] = []
    for agent in agents:
        eligible_player_ids = [
            player_id
            for player_id in round_state.playing_order
            if player_id != agent.id
        ]

        eligible_embeddings = {
            player_id: embeddings_by_player_id[player_id]
            for player_id in eligible_player_ids
        }

        scores_by_player_id = _score_round_candidates(
            round_id=round_state.id,
            turn_id=opening_turn.id,
            playing_order=eligible_player_ids,
            embeddings_by_player_id=eligible_embeddings,
            predictor=predictor,
        )

        target_id = max(
            eligible_player_ids,
            key=lambda player_id: scores_by_player_id[player_id],
        )
        voted_for = (
            HUMAN_PLAYER_NAME
            if target_id == HUMAN_PLAYER_ID
            else agent_names_by_id.get(target_id, target_id)
        )
        votes.append(
            AgentVoteResponse(
                voter_agent_id=agent.id,
                voter_agent_name=agent.name,
                voted_for=voted_for,
                inference_mode="ml_voting",
            )
        )
    return votes


async def _get_or_create_round_embeddings(
    round_id: str,
    phrase_by_player_id: dict[str, str],
    settings: Any,
) -> dict[str, list[float]]:
    round_embeddings = ROUND_EMBEDDINGS.setdefault(round_id, {})
    missing_player_ids = [
        player_id
        for player_id in phrase_by_player_id
        if player_id not in round_embeddings
    ]
    missing_cache_keys = [
        (settings.embedding_model_name, _normalize_embedding_text(phrase_by_player_id[player_id]))
        for player_id in missing_player_ids
    ]

    uncached_player_ids: list[str] = []
    uncached_phrases: list[str] = []
    for player_id, cache_key in zip(missing_player_ids, missing_cache_keys, strict=True):
        cached_embedding = EMBEDDING_CACHE.get(cache_key)
        if cached_embedding is not None:
            round_embeddings[player_id] = cached_embedding
            continue

        uncached_player_ids.append(player_id)
        uncached_phrases.append(phrase_by_player_id[player_id])

    if uncached_phrases:
        client = InferenceClient(settings=settings)
        result = await client.embed(uncached_phrases)
        if len(result.embeddings) != len(uncached_player_ids):
            raise InferenceServiceError("Embedding model returned the wrong number of vectors")

        for player_id, phrase, embedding in zip(
            uncached_player_ids,
            uncached_phrases,
            result.embeddings,
            strict=True,
        ):
            cache_key = (settings.embedding_model_name, _normalize_embedding_text(phrase))
            EMBEDDING_CACHE[cache_key] = embedding
            round_embeddings[player_id] = embedding

    return {
        player_id: embedding
        for player_id, embedding in round_embeddings.items()
        if player_id in phrase_by_player_id
    }



def _normalize_embedding_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _resolve_vote_target(vote_text: str, agents: list[AgentConfig]) -> str | None:
    normalized_vote = vote_text.strip().lower().strip(" .!?,:;\"'")
    if not normalized_vote:
        return None

    if normalized_vote in {"you", "human", "player"}:
        return HUMAN_PLAYER_ID

    for agent in agents:
        if normalized_vote == agent.id.lower() or normalized_vote == agent.name.lower():
            return agent.id

    return None


def _tally_round_votes(
    human_voted_agent_id: str | None,
    agent_votes: list[AgentVoteResponse],
    agents: list[AgentConfig],
) -> tuple[list[VoteCountResponse], str | None]:
    player_names_by_id = _get_player_names_by_id(agents)
    vote_counter: Counter[str] = Counter()
    if human_voted_agent_id is not None:
        vote_counter[human_voted_agent_id] += 1

    for agent_vote in agent_votes:
        target_id = _resolve_vote_target(agent_vote.voted_for, agents)
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


def _get_player_names_by_id(agents: list[AgentConfig]) -> dict[str, str]:
    return {
        HUMAN_PLAYER_ID: HUMAN_PLAYER_NAME,
        **{agent.id: agent.name for agent in agents},
    }


def _build_voting_feature_inputs(
    round_id: str,
    turn_id: str,
    playing_order: list[str],
    embeddings_by_player_id: dict[str, list[float]],
) -> list[VotingFeatureInput]:
    feature_inputs: list[VotingFeatureInput] = []
    for player in playing_order:
        position = playing_order.index(player)
        other_embeddings = [
            embeddings_by_player_id[other_player]
            for other_player in playing_order
            if other_player != player
        ]
        previous_embeddings = [
            embeddings_by_player_id[previous_player]
            for previous_player in playing_order[:position]
        ]
        feature_inputs.append(
            VotingFeatureInput(
                round_id=round_id,
                turn_id=turn_id,
                candidate_turn_position=position,
                candidate_embedding=embeddings_by_player_id[player],
                other_embeddings=other_embeddings,
                previous_embeddings=previous_embeddings,
            )
        )
    return feature_inputs

def _score_round_candidates(
    *,
    round_id: str,
    turn_id: str,
    playing_order: list[str],
    embeddings_by_player_id: dict[str, list[float]],
    predictor: VotingModelPredictor,
) -> dict[str, float]:
    feature_inputs = _build_voting_feature_inputs(
        round_id=round_id,
        turn_id=turn_id,
        playing_order=playing_order,
        embeddings_by_player_id=embeddings_by_player_id,
    )
    feature_matrix = VotingFeatureTransformer().transform(feature_inputs)
    return predictor.score_candidates(playing_order, feature_matrix)
