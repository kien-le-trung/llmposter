from dataclasses import replace
from datetime import UTC, datetime
from random import choice
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.agents import build_clue_system_prompt, get_agent, list_agent_configs
from app.services.inference import InferenceClient, InferenceRequest, InferenceServiceError

router = APIRouter(prefix="/rounds", tags=["rounds"])

HUMAN_PLAYER_ID = "human"
OPENING_PROMPT = "Give your clue now."


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
    user_role: str
    status: str
    turns: list[TurnResponse]
    created_at: datetime


class RoundState(BaseModel):
    id: str
    secret_word: str
    imposter_player_id: str
    status: str
    turns: list[TurnResponse]
    created_at: datetime
    voted_agent_id: str | None = None


class CreateRoundRequest(BaseModel):
    secret_word: str = Field(min_length=1, max_length=80)


class VoteRequest(BaseModel):
    agent_id: str


class VoteResponse(BaseModel):
    voted_agent_id: str
    voted_agent_name: str
    correct: bool
    imposter_was: str


ROUNDS: dict[str, RoundState] = {}


def to_round_response(round_state: RoundState) -> RoundResponse:
    human_is_imposter = round_state.imposter_player_id == HUMAN_PLAYER_ID
    return RoundResponse(
        id=round_state.id,
        visible_word=None if human_is_imposter else round_state.secret_word,
        user_role="imposter" if human_is_imposter else "player",
        status=round_state.status,
        turns=round_state.turns,
        created_at=round_state.created_at,
    )


async def generate_opening_turn(round_state: RoundState) -> TurnResponse:
    client = InferenceClient(settings=settings)
    responses: list[AgentTurnResponse] = []

    try:
        for agent in list_agent_configs():
            agent_knows_word = agent.id != round_state.imposter_player_id
            system_prompt = build_clue_system_prompt(
                round_state.secret_word if agent_knows_word else None
            )
            round_agent = replace(agent, system_prompt=system_prompt)
            result = await client.generate(InferenceRequest(prompt=OPENING_PROMPT, agent=round_agent))
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


@router.post("", response_model=RoundResponse, status_code=201)
async def create_round(payload: CreateRoundRequest) -> RoundResponse:
    agents = list_agent_configs()
    if not agents:
        raise HTTPException(status_code=500, detail="No agents configured")

    player_ids = [HUMAN_PLAYER_ID, *[agent.id for agent in agents]]
    round_state = RoundState(
        id=str(uuid4()),
        secret_word=payload.secret_word,
        imposter_player_id=choice(player_ids),
        status="active",
        turns=[],
        created_at=datetime.now(UTC),
    )
    opening_turn = await generate_opening_turn(round_state)
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
def vote(round_id: str, payload: VoteRequest) -> VoteResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise HTTPException(status_code=404, detail="Unknown round")

    voted_agent = get_agent(payload.agent_id)
    if voted_agent is None:
        raise HTTPException(status_code=404, detail="Unknown agent")

    round_state.voted_agent_id = voted_agent.id
    round_state.status = "complete"

    imposter_agent = get_agent(round_state.imposter_player_id)
    imposter_was = imposter_agent.name if imposter_agent is not None else "You"

    return VoteResponse(
        voted_agent_id=voted_agent.id,
        voted_agent_name=voted_agent.name,
        correct=voted_agent.id == round_state.imposter_player_id,
        imposter_was=imposter_was,
    )
