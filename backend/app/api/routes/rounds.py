import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import random
from random import choice
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import Settings, get_app_settings
from app.db.models import ClueModel, VoteModel
from app.db.session import SessionLocal, get_db
from app.services.agents.clue_generation import build_instruction_clue_user_prompt, clean_clue_response
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
    include_human: bool = True
    human_clue: str | None = None
    voted_agent_id: str | None = None
    completed_clues: dict[str, AgentTurnResponse] = Field(default_factory=dict)


class CreateRoundRequest(BaseModel):
    secret_word: str | None = Field(default=None, min_length=1, max_length=80)
    imposter_hint: str | None = Field(default=None, min_length=1, max_length=160)
    prompt_technique: str | None = Field(default=None, min_length=1, max_length=80)
    include_human: bool = True


class VoteRequest(BaseModel):
    agent_id: str | None = None
    human_clue: str | None = Field(default=None, min_length=1, max_length=200)


class SubmitClueRequest(BaseModel):
    clue: str = Field(min_length=1, max_length=200)


class ClueModelResponse(BaseModel):
    clue: str = Field(min_length=1)


ROUNDS: dict[str, RoundState] = {}
ACTIVE_GENERATION_ROUNDS: set[str] = set()


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


def reveal_available_clues(state: RoundState) -> None:
    opening_turn = get_or_create_opening_turn(state)
    while state.current_player_index < len(state.playing_order):
        player_id = state.playing_order[state.current_player_index]
        clue = state.completed_clues.get(player_id)
        if clue is None:
            state.status = (
                "awaiting_human_clue" if player_id == HUMAN_PLAYER_ID else "generating_clues"
            )
            return
        if not any(response.agent_id == player_id for response in opening_turn.responses):
            opening_turn.responses.append(clue)
        state.current_player_index += 1
    state.status = "ready_to_vote"


async def continue_clue_generation(
    round_id: str,
    agents: list[AgentConfig],
    settings: Settings,
) -> None:
    if round_id in ACTIVE_GENERATION_ROUNDS:
        return
    state = ROUNDS.get(round_id)
    if state is None:
        return
    ACTIVE_GENERATION_ROUNDS.add(round_id)
    try:
        await _run_clue_generation(state, agents, settings)
    finally:
        ACTIVE_GENERATION_ROUNDS.discard(round_id)


async def _run_clue_generation(
    state: RoundState,
    agents: list[AgentConfig],
    settings: Settings,
) -> None:
    with SessionLocal() as db:
        agents_by_id = {agent.id: agent for agent in agents}
        existing_ids = set(state.completed_clues)
        limit = settings.llm_config.max_concurrent_requests
        if limit is None:
            limit = 1 if "localhost:8888" in settings.llm_config.chat_url else len(agents)

        async with InferenceClient(settings, max_concurrent_requests=max(1, limit)) as client:
            tasks: dict[asyncio.Task[AgentTurnResponse], str] = {}
            for agent in agents:
                if agent.id != state.imposter_player_id and agent.id not in existing_ids:
                    task = asyncio.create_task(
                        generate_agent_clue(state, agent, [], settings, client=client)
                    )
                    tasks[task] = agent.id

            imposter_task_started = state.imposter_player_id == HUMAN_PLAYER_ID

            def start_imposter_if_ready() -> None:
                nonlocal imposter_task_started
                if imposter_task_started:
                    return
                imposter_index = state.playing_order.index(state.imposter_player_id)
                prerequisite_ids = state.playing_order[:imposter_index]
                if not all(player_id in state.completed_clues for player_id in prerequisite_ids):
                    return
                imposter = agents_by_id[state.imposter_player_id]
                preceding = [
                    AgentTurnResponse(
                        agent_id=player_id,
                        agent_name=state.player_names_by_id[player_id],
                        agent_response=state.completed_clues[player_id].agent_response,
                        inference_mode="stored",
                    )
                    for player_id in prerequisite_ids
                ]
                task = asyncio.create_task(
                    generate_agent_clue(state, imposter, preceding, settings, client=client)
                )
                tasks[task] = imposter.id
                imposter_task_started = True

            start_imposter_if_ready()
            try:
                while tasks:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        player_id = tasks.pop(task)
                        response = task.result()
                        state.completed_clues[player_id] = response
                        db.add(ClueModel(
                            id=str(uuid4()), round_id=state.id, player_id=player_id,
                            player_name=response.agent_name, clue=response.agent_response,
                            inference_mode=response.inference_mode, sequence=1,
                        ))
                    db.commit()
                    start_imposter_if_ready()
                    reveal_available_clues(state)
            except (InferenceServiceError, KeyError, ValueError):
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                state.status = "generation_failed"
                return

        reveal_available_clues(state)

async def generate_agent_clue(
    round_state: RoundState,
    agent: AgentConfig,
    previous_responses: list[AgentTurnResponse],
    settings: Settings,
    client: InferenceClient | None = None,
) -> AgentTurnResponse:
    inference_client = client or InferenceClient(settings=settings)
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
    structured_result, result = await inference_client.generate_structured(
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
    if payload is not None and payload.secret_word:
        secret_word = payload.secret_word
        imposter_hint = (
            payload.imposter_hint
            if payload.imposter_hint
            else settings.fixed_imposter_hint
        )
        return secret_word, normalize_imposter_hint(imposter_hint)

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

    include_human = payload.include_human if payload is not None else True
    player_ids = [agent.id for agent in agents]
    if include_human:
        player_ids.insert(0, HUMAN_PLAYER_ID)

    playing_order = player_ids.copy()
    random.shuffle(playing_order)

    player_names_by_id = {agent.id: agent.name for agent in agents}
    if include_human:
        player_names_by_id[HUMAN_PLAYER_ID] = HUMAN_PLAYER_NAME

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
        status="generating_clues",
        playing_order=playing_order,
        player_names_by_id=player_names_by_id,
        prompt_technique=normalized_prompt_technique,
        current_player_index=0,
        turns=[],
        created_at=datetime.now(UTC),
        include_human=include_human,
    )
    ROUNDS[round_state.id] = round_state
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
    background_tasks: BackgroundTasks,
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

    human_response = AgentTurnResponse(
        agent_id=HUMAN_PLAYER_ID, agent_name=HUMAN_PLAYER_NAME,
        agent_response=human_clue, inference_mode="human",
    )
    round_state.human_clue = human_clue
    round_state.completed_clues[HUMAN_PLAYER_ID] = human_response
    db.add(ClueModel(
        id=str(uuid4()), round_id=round_id, player_id=HUMAN_PLAYER_ID,
        player_name=HUMAN_PLAYER_NAME, clue=human_clue, inference_mode=None,
        sequence=1,
    ))
    reveal_available_clues(round_state)
    db.commit()
    agents = list_runtime_agent_configs(db, settings)
    background_tasks.add_task(continue_clue_generation, round_id, agents, settings)
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

    voted_agent = (
        get_runtime_agent_config(db, payload.agent_id, settings)
        if payload.agent_id is not None
        else None
    )
    if payload.agent_id is not None and voted_agent is None:
        raise HTTPException(status_code=404, detail="Unknown agent")

    if round_state.status != "ready_to_vote":
        raise HTTPException(status_code=409, detail="Round is not ready for voting")

    round_has_human = HUMAN_PLAYER_ID in round_state.playing_order
    if round_has_human and payload.agent_id is None:
        raise HTTPException(status_code=422, detail="A human vote agent_id is required")

    agents = list_runtime_agent_configs(db, settings)
    try:
        result = await submit_round_vote(
            round_state,
            agents,
            voted_agent,
            payload.human_clue,
            settings,
        )
        names_by_id = get_player_names_by_id_from_round(round_state)
        if HUMAN_PLAYER_ID in round_state.playing_order:
            db.add(VoteModel(
                id=str(uuid4()), round_id=round_id, voter_id=HUMAN_PLAYER_ID,
                voter_name=HUMAN_PLAYER_NAME, voted_for_id=result.voted_agent_id,
                voted_for_name=result.voted_agent_name, raw_vote=None, inference_mode=None,
            ))
        for agent_vote in result.agent_votes:
            normalized_vote = agent_vote.voted_for.strip().lower().strip(" .!?,:;\"'")
            target_id = next(
                (player_id for player_id, name in names_by_id.items()
                 if normalized_vote in {player_id.lower(), name.lower()}),
                None,
            )
            db.add(VoteModel(
                id=str(uuid4()), round_id=round_id, voter_id=agent_vote.voter_agent_id,
                voter_name=agent_vote.voter_agent_name, voted_for_id=target_id,
                voted_for_name=names_by_id.get(target_id) if target_id else None,
                raw_vote=agent_vote.voted_for, inference_mode=agent_vote.inference_mode,
            ))
        db.add(VoteModel(
            id=str(uuid4()), round_id=round_id, voter_id="group", voter_name="Group",
            voted_for_id=result.group_voted_player_id,
            voted_for_name=result.group_voted_player_name, raw_vote=None,
            inference_mode="tally", imposter_won=result.imposter_won,
            round_winner=result.round_winner,
        ))
        db.commit()
        return result
    except VotingStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InferenceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
