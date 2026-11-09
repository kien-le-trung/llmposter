import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
import random
from random import choice
from typing import Literal
from uuid import uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import ClueModel, VoteModel
from app.db.session import SessionLocal
from app.services import latency
from app.services.agents import discussion as chat
from app.services.agents.clue_agent import ClueContext, run_clue
from app.services.agents.strategy_loader import (
    PROMPT_TECHNIQUES,
    normalize_prompt_technique,
)
from app.services.agents.inference import AgentConfig
from app.services.agents.inference import InferenceClient, InferenceServiceError
from app.services.agents.runtime_agents import get_runtime_agent_config, list_runtime_agent_configs
from app.services.voting.voting import VoteResponse, submit_round_vote
from app.services.word_bank import normalize_imposter_hint, select_random_word

from app.services.game.schemas import (
    AgentTurnResponse, TurnResponse, PlayerResponse, RoundResponse,
    CreateRoundRequest, VoteRequest, SubmitClueRequest,
    SubmitDiscussionComment,
)
from app.services.game.state import (
    RoundState, ROUNDS, ACTIVE_GENERATION_ROUNDS, HUMAN_PLAYER_ID,
)

ROUND_PLAYER_NAMES = (
    "Maya", "Alex", "Jordan", "Priya", "Leo", "Sofia", "Noah", "Amara",
    "Ethan", "Isabel", "Omar", "Chloe", "Felix", "Nina", "Luca", "Zoe",
    "Arjun", "Elena", "Theo", "Aisha", "Sam", "Clara", "Miles", "Yuna",
    "Jamie", "Leila", "Oscar", "Iris", "Ravi", "Freya", "Daniel", "Naomi",
)

OPENING_PROMPT = "Give your clue now."


class GameError(Exception):
    """A gameplay failure whose transport representation belongs to the caller."""

    def __init__(
        self,
        kind: Literal["not_found", "invalid_input", "conflict", "configuration", "unavailable"],
        message: str,
    ) -> None:
        super().__init__(message)
        self.kind = kind


def round_state_to_round_response(round_state: RoundState) -> RoundResponse:
    chat.expire_discussion(round_state)
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
        discussion=round_state.discussion,
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
            latency.emit("clue_revealed", round_id=state.id, phase="clues",
                         agent_id=player_id, turn_id=f"clue:{player_id}")
        state.current_player_index += 1
    chat.open_discussion(state)


async def continue_clue_generation(
    round_id: str,
    agents: list[AgentConfig],
    settings: Settings,
    http_client=None,
) -> None:
    if round_id in ACTIVE_GENERATION_ROUNDS:
        return
    state = ROUNDS.get(round_id)
    if state is None:
        return
    ACTIVE_GENERATION_ROUNDS.add(round_id)
    try:
        await _run_clue_generation(state, agents, settings, http_client=http_client)
    finally:
        ACTIVE_GENERATION_ROUNDS.discard(round_id)


async def _run_clue_generation(
    state: RoundState,
    agents: list[AgentConfig],
    settings: Settings,
    http_client=None,
) -> None:
    with SessionLocal() as db:
        agents_by_id = {agent.id: agent for agent in agents}
        existing_ids = set(state.completed_clues)
        limit = settings.llm_config.max_concurrent_requests
        if limit is None:
            limit = 1 if "localhost:8888" in settings.llm_config.chat_url else len(agents)

        client = InferenceClient(settings, max_concurrent_requests=max(1, limit), http_client=http_client)
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
            while tasks: # when there are still tasks to complete
                # return the first completed task(s), process it. This will be called multiple times
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
                with latency.span("clue_persistence", round_id=state.id, phase="clues"):
                    db.commit()
                for task in done:
                    response = task.result()
                    latency.emit("clue_persisted", round_id=state.id, phase="clues",
                                 agent_id=response.agent_id, turn_id=f"clue:{response.agent_id}")
                start_imposter_if_ready()
                reveal_available_clues(state)
                chat.ensure_discussion_tasks(state, agents, settings, http_client=http_client)
        except (InferenceServiceError, KeyError, ValueError):
            for task in tasks:
                task.cancel() # send a cancellation request to all remaining tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            state.status = "generation_failed"
            return

        reveal_available_clues(state)


async def generate_agent_clue(
    round_state: RoundState,
    agent: AgentConfig,
    previous_responses: list[AgentTurnResponse],
    settings: Settings,
    *, client: InferenceClient,
) -> AgentTurnResponse:
    with latency.span("clue_graph", round_id=round_state.id, phase="clues",
                      agent_id=agent.id, turn_id=f"clue:{agent.id}"):
        imposter = agent.id == round_state.imposter_player_id
        context = ClueContext(
            secret_word=None if imposter else round_state.secret_word,
            imposter_hint=round_state.imposter_hint if imposter else None,
            previous_clues=tuple(
                (round_state.player_names_by_id[r.agent_id], r.agent_response)
                for r in previous_responses
            ) if imposter else (),
            prompt_technique=round_state.prompt_technique,
            your_id=agent.id,
            your_name=round_state.player_names_by_id[agent.id],
            players=tuple(
                (pid, round_state.player_names_by_id[pid]) for pid in round_state.playing_order
            ),
        )
        result = await run_clue(context, agent, client)
        return AgentTurnResponse(
            agent_id=agent.id, agent_name=round_state.player_names_by_id[agent.id],
            agent_response=result["clue"], inference_mode=result["inference_mode"],
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
    return dict(round_state.player_names_by_id)


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


async def create_round(
    payload: CreateRoundRequest | None,
    settings: Settings,
    db: Session,
    http_client=None,
    *,
    schedule: Callable[..., None],
) -> RoundResponse:
    agents = list_runtime_agent_configs(db, settings)
    if not agents:
        raise GameError("configuration", "No agents configured")

    include_human = payload.include_human if payload is not None else True
    player_ids = [agent.id for agent in agents]
    if include_human:
        player_ids.insert(0, HUMAN_PLAYER_ID)

    playing_order = player_ids.copy()
    random.shuffle(playing_order)

    name_pool = random.sample(ROUND_PLAYER_NAMES, len(ROUND_PLAYER_NAMES))
    player_names_by_id = {
        player_id: name_pool[index % len(name_pool)] + (
            f" {index // len(name_pool) + 1}" if index >= len(name_pool) else ""
        )
        for index, player_id in enumerate(player_ids)
    }

    secret_word, imposter_hint = select_round_word(payload, settings)
    prompt_technique = (
        payload.prompt_technique.strip()
        if payload is not None and payload.prompt_technique is not None
        else settings.clue_prompt_technique
    )
    normalized_prompt_technique = normalize_prompt_technique(prompt_technique)
    if prompt_technique.lower() not in PROMPT_TECHNIQUES:
        raise GameError(
            "invalid_input",
            (
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
        discussion=chat.DiscussionState(),
        discussion_duration_seconds=settings.discussion_duration_seconds,
    )
    # Store the round state in memory
    ROUNDS[round_state.id] = round_state
    latency.emit("round_created", round_id=round_state.id, phase="clues",
                 playing_order=playing_order, imposter_player_id=round_state.imposter_player_id,
                 discussion_seconds=round_state.discussion_duration_seconds)

    # Let the caller register background work with its own scheduler.
    schedule(
        continue_clue_generation,
        round_state.id,
        agents,
        settings,
        http_client,
    )

    return round_state_to_round_response(round_state)


async def get_round(round_id: str) -> RoundResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise GameError("not_found", "Unknown round")
    return round_state_to_round_response(round_state)


async def submit_clue(
    round_id: str,
    payload: SubmitClueRequest,
    db: Session,
    settings: Settings,
    http_client=None,
    *,
    schedule: Callable[..., None],
) -> RoundResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise GameError("not_found", "Unknown round")

    if HUMAN_PLAYER_ID not in round_state.playing_order:
        raise GameError("conflict", "Round has no human participant")
    chat.expire_discussion(round_state)
    if round_state.status != "awaiting_human_clue":
        raise GameError("conflict", "Round is not waiting for a human clue")

    current_player_id = round_state.playing_order[round_state.current_player_index] if round_state.current_player_index < len(round_state.playing_order) else None
    if current_player_id != HUMAN_PLAYER_ID:
        raise GameError("conflict", "It is not the human player's turn")

    human_clue = payload.clue.strip()
    if not human_clue:
        raise GameError("invalid_input", "Human clue cannot be blank")

    human_response = AgentTurnResponse(
        agent_id=HUMAN_PLAYER_ID, agent_name=round_state.player_names_by_id[HUMAN_PLAYER_ID],
        agent_response=human_clue, inference_mode="human",
    )
    round_state.human_clue = human_clue
    round_state.completed_clues[HUMAN_PLAYER_ID] = human_response
    db.add(ClueModel(
        id=str(uuid4()), round_id=round_id, player_id=HUMAN_PLAYER_ID,
        player_name=round_state.player_names_by_id[HUMAN_PLAYER_ID], clue=human_clue, inference_mode=None,
        sequence=1,
    ))
    reveal_available_clues(round_state)
    db.commit()
    agents = list_runtime_agent_configs(db, settings)
    chat.ensure_discussion_tasks(round_state, agents, settings, http_client=http_client)
    schedule(continue_clue_generation, round_id, agents, settings, http_client)
    return round_state_to_round_response(round_state)


async def vote(
    round_id: str,
    payload: VoteRequest,
    db: Session,
    settings: Settings,
    http_client=None,
) -> VoteResponse:
    round_state = ROUNDS.get(round_id)
    if round_state is None:
        raise GameError("not_found", "Unknown round")

    voted_agent = (
        get_runtime_agent_config(db, payload.agent_id, settings)
        if payload.agent_id is not None
        else None
    )
    if payload.agent_id is not None and (
        voted_agent is None or payload.agent_id not in round_state.playing_order
    ):
        raise GameError("not_found", "Unknown agent")

    chat.expire_discussion(round_state)
    if round_state.status != "ready_to_vote":
        raise GameError("conflict", "Round is not ready for voting")

    round_has_human = HUMAN_PLAYER_ID in round_state.playing_order
    if not round_has_human and payload.agent_id is not None:
        raise GameError("invalid_input", "AI-only rounds do not accept a human vote")
    if round_has_human and payload.agent_id is None:
        raise GameError("invalid_input", "A human vote agent_id is required")

    agents = list_runtime_agent_configs(db, settings)
    async with round_state.discussion._lock:
        if round_state.discussion._voting or round_state.status != "ready_to_vote":
            raise GameError("conflict", "Voting already in progress or complete")
        round_state.discussion._voting = True
    try:
        with latency.span("voting_graph", round_id=round_id, phase="voting"):
            result = await submit_round_vote(round_state, agents, voted_agent, settings, http_client=http_client)
        names_by_id = get_player_names_by_id_from_round(round_state)
        if HUMAN_PLAYER_ID in round_state.playing_order:
            db.add(VoteModel(
                id=str(uuid4()), round_id=round_id, voter_id=HUMAN_PLAYER_ID,
                voter_name=names_by_id[HUMAN_PLAYER_ID], voted_for_id=result.voted_agent_id,
                voted_for_name=result.voted_agent_name, raw_vote=None, inference_mode=None,
            ))
        for agent_vote in result.agent_votes:
            target_id = agent_vote.voted_for if agent_vote.voted_for in names_by_id else None
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
        with latency.span("vote_persistence", round_id=round_id, phase="voting"):
            db.commit()
        round_state.voted_agent_id = result.voted_agent_id
        round_state.status = "complete"
        latency.emit("round_complete", round_id=round_id, phase="voting")
        return result
    except InferenceServiceError as exc:
        raise GameError("unavailable", "Voting inference failed; please retry") from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise GameError("unavailable", "Vote persistence failed; please retry") from exc
    finally:
        round_state.discussion._voting = False


async def submit_discussion_comment(
    round_id: str,
    payload: SubmitDiscussionComment,
    db: Session,
) -> RoundResponse:
    state = ROUNDS.get(round_id)
    if state is None:
        raise GameError("not_found", "Unknown round")
    if HUMAN_PLAYER_ID not in state.playing_order:
        raise GameError("conflict", "Round has no human participant")
    async with state.discussion._lock:
        chat.expire_discussion(state)
        if state.status != "discussion":
            raise GameError("conflict", "Discussion is closed")
        if not payload.text.strip():
            raise GameError("invalid_input", "Comment cannot be blank")
        chat.publish_comment(state, HUMAN_PLAYER_ID, payload.text.strip(), "human", db)
    return round_state_to_round_response(state)
