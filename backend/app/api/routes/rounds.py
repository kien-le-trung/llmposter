from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from random import choice
from random import choice as random_choice
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.services.agents import build_clue_system_prompt, build_vote_system_prompt
from app.services.inference import AgentConfig
from app.services.inference import InferenceClient, InferenceRequest, InferenceServiceError
from app.services.runtime_agents import get_runtime_agent_config, list_runtime_agent_configs

router = APIRouter(prefix="/rounds", tags=["rounds"])

HUMAN_PLAYER_ID = "human"
HUMAN_PLAYER_NAME = "You"
OPENING_PROMPT = "Give your clue now."
WORD_BANK: list[tuple[str, str]] = [
    ("apple", "Fruit, orchard, or red"),
    ("bridge", "Crossing, river, or structure"),
    ("camera", "Photos, lens, or memories"),
    ("desert", "Sand, heat, or dunes"),
    ("forest", "Trees, shade, or wilderness"),
    ("guitar", "Music, strings, or stage"),
    ("island", "Water, coast, or isolation"),
    ("library", "Books, shelves, or quiet"),
    ("mountain", "Peak, climb, or snow"),
    ("piano", "Music, keys, or concert"),
    ("rocket", "Launch, space, or engines"),
    ("satellite", "Space, signals, or orbit"),
    ("theater", "Stage, actors, or curtains"),
    ("volcano", "Lava, ash, or eruption"),
    ("window", "Glass, view, or sunlight"),
]


class AgentTurnResponse(BaseModel):
    agent_id: str
    agent_name: str
    agent_response: str
    inference_mode: str


class TurnResponse(BaseModel):
    id: str
    sequence: int
    user_prompt: str
    responses: list[AgentTurnResponse]
    created_at: datetime


class RoundResponse(BaseModel):
    id: str
    visible_word: str | None
    imposter_hint: str | None
    user_role: str
    status: str
    turns: list[TurnResponse]
    created_at: datetime


class RoundState(BaseModel):
    id: str
    secret_word: str
    imposter_hint: str
    imposter_player_id: str
    status: str
    turns: list[TurnResponse]
    created_at: datetime
    voted_agent_id: str | None = None


class CreateRoundRequest(BaseModel):
    secret_word: str | None = Field(default=None, min_length=1, max_length=80)
    imposter_hint: str | None = Field(default=None, min_length=1, max_length=160)


class VoteRequest(BaseModel):
    agent_id: str
    human_clue: str = Field(min_length=1, max_length=200)


class AgentVoteResponse(BaseModel):
    voter_agent_id: str
    voter_agent_name: str
    voted_for: str
    inference_mode: str


class VoteCountResponse(BaseModel):
    player_id: str
    player_name: str
    votes: int


class VoteResponse(BaseModel):
    voted_agent_id: str
    voted_agent_name: str
    secret_word: str
    imposter_was: str
    agent_votes: list[AgentVoteResponse]
    vote_counts: list[VoteCountResponse]
    group_voted_player_id: str | None
    group_voted_player_name: str | None
    imposter_won: bool
    round_winner: str


ROUNDS: dict[str, RoundState] = {}


def to_round_response(round_state: RoundState) -> RoundResponse:
    human_is_imposter = round_state.imposter_player_id == HUMAN_PLAYER_ID
    return RoundResponse(
        id=round_state.id,
        visible_word=None if human_is_imposter else round_state.secret_word,
        imposter_hint=round_state.imposter_hint if human_is_imposter else None,
        user_role="imposter" if human_is_imposter else "player",
        status=round_state.status,
        turns=round_state.turns,
        created_at=round_state.created_at,
    )


async def generate_opening_turn(
    round_state: RoundState,
    agents: list[AgentConfig],
) -> TurnResponse:
    client = InferenceClient(settings=settings)
    responses: list[AgentTurnResponse] = []

    try:
        for agent in agents:
            agent_knows_word = agent.id != round_state.imposter_player_id
            system_prompt = build_clue_system_prompt(
                round_state.secret_word if agent_knows_word else None,
                None if agent_knows_word else round_state.imposter_hint,
            )
            round_agent = replace(agent, system_prompt=system_prompt)
            result = await client.generate(
                InferenceRequest(prompt=build_opening_prompt(responses), agent=round_agent)
            )
            responses.append(
                AgentTurnResponse(
                    agent_id=agent.id,
                    agent_name=agent.name,
                    agent_response=result.text,
                    inference_mode=result.inference_mode,
                )
            )
    except InferenceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return TurnResponse(
        id=str(uuid4()),
        sequence=1,
        user_prompt=OPENING_PROMPT,
        responses=responses,
        created_at=datetime.now(UTC),
    )


def build_opening_prompt(previous_responses: list[AgentTurnResponse]) -> str:
    if not previous_responses:
        return OPENING_PROMPT

    clue_lines = [
        f"{response.agent_name}: {response.agent_response}" for response in previous_responses
    ]
    return (
        "Previous clues from this round:\n"
        + "\n".join(clue_lines)
        + f"\n\n{OPENING_PROMPT}"
    )


def build_vote_prompt(
    voter_agent_id: str,
    human_clue: str,
    opening_turn: TurnResponse,
) -> str:
    clue_lines = [f"{HUMAN_PLAYER_NAME}: {human_clue}"]
    for response in opening_turn.responses:
        if response.agent_id != voter_agent_id:
            clue_lines.append(f"{response.agent_name}: {response.agent_response}")

    return "Clues from the other players:\n" + "\n".join(clue_lines)


async def generate_agent_votes(
    round_state: RoundState,
    agents: list[AgentConfig],
    human_clue: str,
) -> list[AgentVoteResponse]:
    if not round_state.turns:
        return []

    client = InferenceClient(settings=settings)
    opening_turn = round_state.turns[0]
    agent_names_by_id = {agent.id: agent.name for agent in agents}
    votes: list[AgentVoteResponse] = []

    try:
        for agent_index, agent in enumerate(agents):
            other_agent_names = [
                candidate_name
                for candidate_id, candidate_name in agent_names_by_id.items()
                if candidate_id != agent.id
            ]
            rotation_index = agent_index % len(other_agent_names) if other_agent_names else 0
            rotated_agent_names = (
                other_agent_names[rotation_index:] + other_agent_names[:rotation_index]
            )
            candidate_names = [*rotated_agent_names, HUMAN_PLAYER_NAME]
            vote_agent = replace(
                agent,
                system_prompt=build_vote_system_prompt(agent.name, candidate_names),
                max_tokens=12,
                temperature=0.2,
            )
            result = await client.generate(
                InferenceRequest(
                    prompt=build_vote_prompt(agent.id, human_clue, opening_turn),
                    agent=vote_agent,
                )
            )
            votes.append(
                AgentVoteResponse(
                    voter_agent_id=agent.id,
                    voter_agent_name=agent.name,
                    voted_for=result.text.strip(),
                    inference_mode=result.inference_mode,
                )
            )
    except InferenceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return votes


def get_player_names_by_id(agents: list[AgentConfig]) -> dict[str, str]:
    return {
        HUMAN_PLAYER_ID: HUMAN_PLAYER_NAME,
        **{agent.id: agent.name for agent in agents},
    }


def resolve_vote_target(vote_text: str, agents: list[AgentConfig]) -> str | None:
    normalized_vote = vote_text.strip().lower().strip(" .!?,:;\"'")
    if not normalized_vote:
        return None

    if normalized_vote in {"you", "human", "player"}:
        return HUMAN_PLAYER_ID

    for agent in agents:
        if normalized_vote == agent.id.lower() or normalized_vote == agent.name.lower():
            return agent.id

    return None


def tally_round_votes(
    human_voted_agent_id: str,
    agent_votes: list[AgentVoteResponse],
    agents: list[AgentConfig],
) -> tuple[list[VoteCountResponse], str | None]:
    player_names_by_id = get_player_names_by_id(agents)
    vote_counter: Counter[str] = Counter([human_voted_agent_id])

    for agent_vote in agent_votes:
        target_id = resolve_vote_target(agent_vote.voted_for, agents)
        if target_id is not None:
            vote_counter[target_id] += 1

    if not vote_counter:
        return [], None

    highest_vote_total = max(vote_counter.values())
    leading_player_ids = [
        player_id for player_id, vote_total in vote_counter.items() if vote_total == highest_vote_total
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


def select_round_word(payload: CreateRoundRequest | None) -> tuple[str, str]:
    if settings.word_selection_mode == "fixed":
        return (
            payload.secret_word if payload and payload.secret_word else settings.fixed_secret_word,
            payload.imposter_hint if payload and payload.imposter_hint else settings.fixed_imposter_hint,
        )

    return random_choice(WORD_BANK)


@router.post("", response_model=RoundResponse, status_code=201)
async def create_round(
    payload: CreateRoundRequest | None = None,
    db: Session = Depends(get_db),
) -> RoundResponse:
    agents = list_runtime_agent_configs(db)
    if not agents:
        raise HTTPException(status_code=500, detail="No agents configured")

    player_ids = [HUMAN_PLAYER_ID, *[agent.id for agent in agents]]
    secret_word, imposter_hint = select_round_word(payload)
    round_state = RoundState(
        id=str(uuid4()),
        secret_word=secret_word,
        imposter_hint=imposter_hint,
        imposter_player_id=choice(player_ids),
        status="active",
        turns=[],
        created_at=datetime.now(UTC),
    )
    opening_turn = await generate_opening_turn(round_state, agents)
    round_state.turns.append(opening_turn)
    ROUNDS[round_state.id] = round_state
    return to_round_response(round_state)


@router.get("/{round_id}", response_model=RoundResponse)
def get_round(round_id: str) -> RoundResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise HTTPException(status_code=404, detail="Unknown round")

    return to_round_response(round_state)


@router.post("/{round_id}/vote", response_model=VoteResponse)
async def vote(
    round_id: str,
    payload: VoteRequest,
    db: Session = Depends(get_db),
) -> VoteResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise HTTPException(status_code=404, detail="Unknown round")

    voted_agent = get_runtime_agent_config(db, payload.agent_id)
    if voted_agent is None:
        raise HTTPException(status_code=404, detail="Unknown agent")

    agents = list_runtime_agent_configs(db)
    agent_votes = await generate_agent_votes(round_state, agents, payload.human_clue)
    vote_counts, group_voted_player_id = tally_round_votes(voted_agent.id, agent_votes, agents)

    round_state.voted_agent_id = voted_agent.id
    round_state.status = "complete"

    imposter_agent = get_runtime_agent_config(db, round_state.imposter_player_id)
    imposter_was = imposter_agent.name if imposter_agent is not None else "You"
    player_names_by_id = get_player_names_by_id(agents)
    group_voted_player_name = (
        player_names_by_id.get(group_voted_player_id) if group_voted_player_id is not None else None
    )
    imposter_won = group_voted_player_id != round_state.imposter_player_id

    return VoteResponse(
        voted_agent_id=voted_agent.id,
        voted_agent_name=voted_agent.name,
        secret_word=round_state.secret_word,
        imposter_was=imposter_was,
        agent_votes=agent_votes,
        vote_counts=vote_counts,
        group_voted_player_id=group_voted_player_id,
        group_voted_player_name=group_voted_player_name,
        imposter_won=imposter_won,
        round_winner="imposter" if imposter_won else "players",
    )
