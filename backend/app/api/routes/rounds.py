from dataclasses import replace
from datetime import UTC, datetime
import random
from random import choice
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import Settings, get_app_settings
from app.db.session import get_db
from app.services.agents.clue_generation import (
    build_instruction_batched_clue_user_prompt,
    build_instruction_clue_user_prompt,
    clean_batched_clue_response,
    clean_clue_response,
)
from app.services.agents.strategy_loader import (
    PROMPT_TECHNIQUES,
    assign_imposter_clue_strategy,
    assign_non_imposter_clue_strategy,
    normalize_prompt_technique,
)
from app.services.agents.inference import AgentConfig
from app.services.agents.inference import InferenceClient, InferenceRequest, InferenceServiceError
from app.services.agents.runtime_agents import get_runtime_agent_config, list_runtime_agent_configs
from app.services.voting.voting import VoteResponse, VotingStateError, submit_round_vote
from app.services.word_bank import normalize_imposter_hint, select_random_word

router = APIRouter(prefix="/rounds", tags=["rounds"])

HUMAN_PLAYER_ID = "human"
HUMAN_PLAYER_NAME = "You"
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


class PlayerResponse(BaseModel):
    id: str
    name: str
    kind: str


class RoundResponse(BaseModel):
    id: str
    visible_word: str | None
    imposter_hint: str | None
    user_role: str
    prompt_technique: str
    status: str
    playing_order: list[PlayerResponse]
    current_player_id: str | None
    current_player_name: str | None
    turns: list[TurnResponse]
    created_at: datetime


class RoundState(BaseModel):
    id: str
    secret_word: str
    imposter_hint: str
    imposter_player_id: str
    status: str
    playing_order: list[str]
    player_names_by_id: dict[str, str]
    prompt_technique: str
    current_player_index: int
    turns: list[TurnResponse]
    created_at: datetime
    human_clue: str | None = None
    voted_agent_id: str | None = None


class CreateRoundRequest(BaseModel):
    secret_word: str | None = Field(default=None, min_length=1, max_length=80)
    imposter_hint: str | None = Field(default=None, min_length=1, max_length=160)
    prompt_technique: str | None = Field(default=None, min_length=1, max_length=80)


class VoteRequest(BaseModel):
    agent_id: str
    human_clue: str | None = Field(default=None, min_length=1, max_length=200)


class SubmitClueRequest(BaseModel):
    clue: str = Field(min_length=1, max_length=200)


class ClueModelResponse(BaseModel):
    clue: str = Field(min_length=1)


class BatchedClueModelResponse(BaseModel):
    clues: dict[str, str] = Field(min_length=1)


ROUNDS: dict[str, RoundState] = {}


def to_round_response(round_state: RoundState) -> RoundResponse:
    human_is_imposter = round_state.imposter_player_id == HUMAN_PLAYER_ID
    player_names_by_id = get_player_names_by_id_from_round(round_state)
    current_player_id = round_state.playing_order[round_state.current_player_index] if round_state.current_player_index < len(round_state.playing_order) else None
    return RoundResponse(
        id=round_state.id,
        visible_word=None if human_is_imposter else round_state.secret_word,
        imposter_hint=round_state.imposter_hint if human_is_imposter else None,
        user_role="imposter" if human_is_imposter else "player",
        prompt_technique=round_state.prompt_technique,
        status=round_state.status,
        playing_order=[
            PlayerResponse(
                id=player_id,
                name=player_names_by_id.get(player_id, player_id),
                kind="human" if player_id == HUMAN_PLAYER_ID else "agent",
            )
            for player_id in round_state.playing_order
        ],
        current_player_id=current_player_id,
        current_player_name=(
            player_names_by_id.get(current_player_id) if current_player_id is not None else None
        ),
        turns=round_state.turns,
        created_at=round_state.created_at,
    )


async def advance_clue_generation(
    round_state: RoundState,
    agents: list[AgentConfig],
    settings: Settings,
    stop_after_first_agent: bool = False,
) -> None:
    agents_by_id = {agent.id: agent for agent in agents}
    opening_turn = get_or_create_opening_turn(round_state)

    try:
        while round_state.current_player_index < len(round_state.playing_order):
            player_id = round_state.playing_order[round_state.current_player_index]
            if player_id == HUMAN_PLAYER_ID:
                if not append_human_clue_if_available(round_state, opening_turn):
                    round_state.status = "awaiting_human_clue"
                    return

                round_state.current_player_index += 1
                continue

            segment = collect_agent_generation_segment(round_state, agents_by_id)
            if not segment:
                round_state.current_player_index += 1
                continue

            if stop_after_first_agent and not opening_turn.responses:
                response = await generate_agent_clue(
                    round_state,
                    segment[0],
                    opening_turn.responses,
                    settings,
                )
                opening_turn.responses.append(response)
                round_state.current_player_index += 1
                round_state.status = "generating_clues"
                return

            responses = await generate_agent_segment(
                round_state,
                segment,
                opening_turn.responses,
                settings,
            )
            opening_turn.responses.extend(responses)
            round_state.current_player_index += len(segment)
    except InferenceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    round_state.status = "ready_to_vote"


async def continue_clue_generation(
    round_id: str,
    agents: list[AgentConfig],
    settings: Settings,
) -> None:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        return

    try:
        await advance_clue_generation(round_state, agents, settings)
    except HTTPException:
        round_state.status = "generation_failed"


def append_human_clue_if_available(
    round_state: RoundState,
    opening_turn: TurnResponse,
) -> bool:
    if round_state.human_clue is None:
        return False

    if not any(response.agent_id == HUMAN_PLAYER_ID for response in opening_turn.responses):
        opening_turn.responses.append(
            AgentTurnResponse(
                agent_id=HUMAN_PLAYER_ID,
                agent_name=HUMAN_PLAYER_NAME,
                agent_response=round_state.human_clue,
                inference_mode="human",
            )
        )

    return True


def collect_agent_generation_segment(
    round_state: RoundState,
    agents_by_id: dict[str, AgentConfig],
) -> list[AgentConfig]:
    segment: list[AgentConfig] = []
    index = round_state.current_player_index

    while index < len(round_state.playing_order):
        player_id = round_state.playing_order[index]
        if player_id == HUMAN_PLAYER_ID:
            break

        agent = agents_by_id.get(player_id)
        if agent is not None:
            segment.append(agent)

        index += 1

    return segment


async def generate_agent_segment(
    round_state: RoundState,
    agents: list[AgentConfig],
    previous_responses: list[AgentTurnResponse],
    settings: Settings,
) -> list[AgentTurnResponse]:
    if not agents:
        return []

    generated_by_agent_id: dict[str, AgentTurnResponse] = {}
    working_previous_responses = previous_responses.copy()
    first_player_is_human = round_state.playing_order[0] == HUMAN_PLAYER_ID

    if not first_player_is_human and not previous_responses:
        first_agent = agents[0]
        first_response = await generate_agent_clue(
            round_state,
            first_agent,
            working_previous_responses,
            settings,
        )
        generated_by_agent_id[first_agent.id] = first_response
        working_previous_responses.append(first_response)

    remaining_non_imposters = [
        agent
        for agent in agents
        if agent.id != round_state.imposter_player_id and agent.id not in generated_by_agent_id
    ]
    if remaining_non_imposters:
        if settings.batch_prompting:
            non_imposter_responses = await generate_non_imposter_agent_batch(
                round_state,
                remaining_non_imposters,
                working_previous_responses,
                settings,
            )
        else:
            non_imposter_responses = []
            for agent in remaining_non_imposters:
                response = await generate_agent_clue(
                    round_state,
                    agent,
                    working_previous_responses,
                    settings,
                )
                non_imposter_responses.append(response)
                working_previous_responses.append(response)
        for response in non_imposter_responses:
            generated_by_agent_id[response.agent_id] = response
        if settings.batch_prompting:
            working_previous_responses.extend(non_imposter_responses)

    imposter_agent = next(
        (
            agent
            for agent in agents
            if agent.id == round_state.imposter_player_id and agent.id not in generated_by_agent_id
        ),
        None,
    )
    if imposter_agent is not None:
        imposter_response = await generate_agent_clue(
            round_state,
            imposter_agent,
            working_previous_responses,
            settings,
        )
        generated_by_agent_id[imposter_agent.id] = imposter_response

    return [
        generated_by_agent_id[agent.id]
        for agent in agents
        if agent.id in generated_by_agent_id
    ]


async def generate_agent_clue(
    round_state: RoundState,
    agent: AgentConfig,
    previous_responses: list[AgentTurnResponse],
    settings: Settings,
) -> AgentTurnResponse:
    client = InferenceClient(settings=settings)
    agent_is_imposter = agent.id == round_state.imposter_player_id
    secret_word = None if agent_is_imposter else round_state.secret_word
    system_prompt = "You write 2 to 5 word clues for an imposter word game. Return valid JSON only. If you know the secret word, do not use it or define it."
    max_tokens = 24
    prompt = build_instruction_clue_user_prompt(
        secret_word,
        round_state.imposter_hint if agent_is_imposter else None,
        [(response.agent_name, response.agent_response) for response in previous_responses],
        (
            assign_imposter_clue_strategy(round_state.prompt_technique)
            if agent_is_imposter
            else assign_non_imposter_clue_strategy(round_state.prompt_technique)
        )
    )
    round_agent = replace(agent, system_prompt=system_prompt, max_tokens=max_tokens)
    structured_result, result = await client.generate_structured(
        InferenceRequest(prompt=prompt, agent=round_agent),
        ClueModelResponse,
    )
    clue_text = clean_clue_response(
        structured_result.clue,
        secret_word,
        round_state.imposter_hint,
    )
    return AgentTurnResponse(
        agent_id=agent.id,
        agent_name=agent.name,
        agent_response=clue_text,
        inference_mode=result.inference_mode,
    )


async def generate_non_imposter_agent_batch(
    round_state: RoundState,
    agents: list[AgentConfig],
    previous_responses: list[AgentTurnResponse],
    settings: Settings,
) -> list[AgentTurnResponse]:
    player_names = [agent.name for agent in agents]
    client = InferenceClient(settings=settings)
    prompt = build_instruction_batched_clue_user_prompt(
        round_state.secret_word,
        player_names,
        {
            player_name: assign_non_imposter_clue_strategy(round_state.prompt_technique)
            for player_name in player_names
        },
        [(response.agent_name, response.agent_response) for response in previous_responses]
    )
    batch_agent = replace(
        agents[0],
        id="non_imposter_batch",
        name="Non-imposter batch",
        system_prompt="You write 2 to 5 word clues for several players in an imposter word game. Return valid JSON only. Do not use or define the secret word.",
        max_tokens=max(48, len(agents) * 24),
    )
    structured_result, result = await client.generate_structured(
        InferenceRequest(prompt=prompt, agent=batch_agent),
        BatchedClueModelResponse,
        validate=lambda response: validate_batched_clues_response(response, player_names),
    )
    clues_by_player_name = clean_batched_clue_response(
        structured_result.clues,
        player_names,
        round_state.secret_word,
        round_state.imposter_hint,
    )
    return [
        AgentTurnResponse(
            agent_id=agent.id,
            agent_name=agent.name,
            agent_response=clues_by_player_name[agent.name],
            inference_mode=result.inference_mode,
        )
        for agent in agents
    ]


def validate_batched_clues_response(
    response: BatchedClueModelResponse,
    player_names: list[str],
) -> None:
    missing_players = [
        player_name for player_name in player_names if player_name not in response.clues
    ]
    if missing_players:
        raise ValueError(f"Missing clues for players: {', '.join(missing_players)}")


def get_or_create_opening_turn(round_state: RoundState) -> TurnResponse:
    if round_state.turns:
        return round_state.turns[0]

    opening_turn = TurnResponse(
        id=str(uuid4()),
        sequence=1,
        user_prompt=OPENING_PROMPT,
        responses=[],
        created_at=datetime.now(UTC),
    )
    round_state.turns.append(opening_turn)
    return opening_turn


def get_player_names_by_id_from_round(round_state: RoundState) -> dict[str, str]:
    names_by_id = dict(round_state.player_names_by_id)
    names_by_id.setdefault(HUMAN_PLAYER_ID, HUMAN_PLAYER_NAME)
    if round_state.turns:
        names_by_id.update(
            {
                response.agent_id: response.agent_name
                for response in round_state.turns[0].responses
            }
        )
    return names_by_id


def select_round_word(
    payload: CreateRoundRequest | None,
    settings: Settings,
) -> tuple[str, str]:
    if settings.word_selection_mode == "fixed":
        secret_word = payload.secret_word if payload and payload.secret_word else settings.fixed_secret_word
        imposter_hint = payload.imposter_hint if payload and payload.imposter_hint else settings.fixed_imposter_hint
        return secret_word, normalize_imposter_hint(imposter_hint)

    secret_word, imposter_hint = select_random_word()
    return secret_word, normalize_imposter_hint(imposter_hint)


@router.post("", response_model=RoundResponse, status_code=201)
async def create_round(
    background_tasks: BackgroundTasks,
    payload: CreateRoundRequest | None = None,
    settings: Settings = Depends(get_app_settings),
    db: Session = Depends(get_db),
) -> RoundResponse:
    agents = list_runtime_agent_configs(db, settings)
    if not agents:
        raise HTTPException(status_code=500, detail="No agents configured")

    player_ids = [HUMAN_PLAYER_ID, *[agent.id for agent in agents]]
    playing_order = player_ids.copy()
    random.shuffle(playing_order)
    player_names_by_id = {
        HUMAN_PLAYER_ID: HUMAN_PLAYER_NAME,
        **{agent.id: agent.name for agent in agents},
    }
    secret_word, imposter_hint = select_round_word(payload, settings)
    prompt_technique = (
        payload.prompt_technique.strip()
        if payload is not None and payload.prompt_technique is not None
        else settings.clue_prompt_technique
    )
    normalized_prompt_technique = normalize_prompt_technique(prompt_technique)
    if prompt_technique.lower() not in PROMPT_TECHNIQUES:
        raise HTTPException(
            status_code=422,
            detail=(
                "Unknown prompt technique. Allowed values: "
                f"{', '.join(PROMPT_TECHNIQUES)}"
            ),
        )
    round_state = RoundState(
        id=str(uuid4()),
        secret_word=secret_word,
        imposter_hint=imposter_hint,
        imposter_player_id=choice(player_ids),
        status="active",
        playing_order=playing_order,
        player_names_by_id=player_names_by_id,
        prompt_technique=normalized_prompt_technique,
        current_player_index=0,
        turns=[],
        created_at=datetime.now(UTC),
    )
    ROUNDS[round_state.id] = round_state
    await advance_clue_generation(
        round_state,
        agents,
        settings,
        stop_after_first_agent=True,
    )
    if round_state.status == "generating_clues":
        background_tasks.add_task(continue_clue_generation, round_state.id, agents, settings)
    return to_round_response(round_state)


@router.get("/{round_id}", response_model=RoundResponse)
def get_round(round_id: str) -> RoundResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise HTTPException(status_code=404, detail="Unknown round")

    return to_round_response(round_state)


@router.post("/{round_id}/clue", response_model=RoundResponse)
async def submit_clue(
    round_id: str,
    payload: SubmitClueRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> RoundResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise HTTPException(status_code=404, detail="Unknown round")

    if round_state.status != "awaiting_human_clue":
        raise HTTPException(status_code=409, detail="Round is not waiting for a human clue")

    current_player_id = round_state.playing_order[round_state.current_player_index] if round_state.current_player_index < len(round_state.playing_order) else None
    if current_player_id != HUMAN_PLAYER_ID:
        raise HTTPException(status_code=409, detail="It is not the human player's turn")

    human_clue = payload.clue.strip()
    if not human_clue:
        raise HTTPException(status_code=422, detail="Human clue cannot be blank")

    round_state.human_clue = human_clue
    agents = list_runtime_agent_configs(db, settings)
    await advance_clue_generation(round_state, agents, settings)

    return to_round_response(round_state)


@router.post("/{round_id}/vote", response_model=VoteResponse)
async def vote(
    round_id: str,
    payload: VoteRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> VoteResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise HTTPException(status_code=404, detail="Unknown round")

    voted_agent = get_runtime_agent_config(db, payload.agent_id, settings)
    if voted_agent is None:
        raise HTTPException(status_code=404, detail="Unknown agent")
    if round_state.status != "ready_to_vote":
        raise HTTPException(status_code=409, detail="Round is not ready for voting")

    agents = list_runtime_agent_configs(db, settings)
    try:
        return await submit_round_vote(
            round_state,
            agents,
            voted_agent,
            payload.human_clue,
            settings,
        )
    except VotingStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InferenceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
