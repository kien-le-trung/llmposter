"""Private single-player clue graph; persistence and scheduling belong to the engine."""

import json
import re
from dataclasses import dataclass, replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import BaseModel, Field

from app.services.agents.inference import AgentConfig, InferenceClient, InferenceRequest
from app.services.agents.strategy_loader import (
    assign_imposter_clue_strategy, assign_non_imposter_clue_strategy,
)


class ClueDecision(BaseModel):
    clue: str = Field(min_length=1)


@dataclass(frozen=True)
class ClueContext:
    secret_word: str | None
    imposter_hint: str | None
    previous_clues: tuple[tuple[str, str], ...]
    prompt_technique: str
    your_id: str = ""
    your_name: str = ""
    players: tuple[tuple[str, str], ...] = ()


class ClueInput(TypedDict):
    context: ClueContext
    agent: AgentConfig
    client: InferenceClient


class ClueOutput(TypedDict):
    clue: str
    inference_mode: str


class ClueState(ClueInput, ClueOutput, total=False):
    strategy: dict[str, str]
    prompt: str
    raw_clue: str


async def prepare_context(state: ClueState):
    context = state["context"]
    # Rebuild immutable input and enforce role-specific knowledge boundaries.
    return {"context": ClueContext(
        secret_word=context.secret_word,
        imposter_hint=context.imposter_hint if context.secret_word is None else None,
        previous_clues=tuple(tuple(clue) for clue in context.previous_clues)
        if context.secret_word is None else (),
        prompt_technique=context.prompt_technique,
        your_id=context.your_id, your_name=context.your_name, players=context.players,
    )}


async def select_strategy(state: ClueState):
    context = state["context"]
    strategy = (
        assign_imposter_clue_strategy(context.prompt_technique, len(context.previous_clues))
        if context.secret_word is None
        else assign_non_imposter_clue_strategy(context.prompt_technique)
    )
    return {"strategy": strategy}


async def build_prompt(state: ClueState):
    context = state["context"]
    identity = json.dumps({
        "your_id": context.your_id, "your_name": context.your_name,
        "players": [{"id": pid, "name": name} for pid, name in context.players],
    })
    return {"prompt": "Use round names in dialogue and IDs in structured references.\n" + identity + "\n" + build_instruction_clue_user_prompt(
        context.secret_word, context.imposter_hint, list(context.previous_clues), state["strategy"],
    )}


async def generate_clue(state: ClueState):
    agent = replace(state["agent"], system_prompt=(
        "You write 2 to 5 word clues for an imposter word game. Return valid JSON only. "
        "If you know the secret word, do not use it or define it."
    ), max_tokens=24)
    decision, result = await state["client"].generate_structured(
        InferenceRequest(prompt=state["prompt"], agent=agent), ClueDecision,
    )
    return {"raw_clue": decision.clue, "inference_mode": result.inference_mode}


async def normalize_output(state: ClueState):
    context = state["context"]
    return {"clue": clean_clue_response(
        state["raw_clue"], context.secret_word, context.imposter_hint,
    )}


def build_graph():
    graph = StateGraph(ClueState, input_schema=ClueInput, output_schema=ClueOutput)
    nodes = [prepare_context, select_strategy, build_prompt, generate_clue, normalize_output]
    previous = START
    for node in nodes:
        graph.add_node(node.__name__, node)
        graph.add_edge(previous, node.__name__)
        previous = node.__name__
    graph.add_edge(previous, END)
    return graph.compile()


CLUE_GRAPH = build_graph()


async def run_clue(context: ClueContext, agent: AgentConfig, client: InferenceClient):
    with tracing_context(enabled=False):
        return await CLUE_GRAPH.ainvoke({"context": context, "agent": agent, "client": client})


def build_instruction_clue_user_prompt(
    secret_word: str | None,
    imposter_hint: str | None,
    previous_clues: list[tuple[str, str]],
    strategy: dict[str, str],
) -> str:
    if not previous_clues:
        previous_clue_block = "Previous clues: none\n"
    else:
        previous_clue_block = "Previous clues:\n" + "\n".join(
            f"{player_name}: {clue}" for player_name, clue in previous_clues
        ) + "\n"

    if secret_word is None:
        strategy_prompt = strategy["prompt"]
        return (
            "You are the imposter. You do not know the secret word.\n"
            f"Hint: {imposter_hint or 'common everyday thing'}\n"
            f"{previous_clue_block}"
            f"{strategy_prompt}"
            "Constraints:\n"
            "- Return JSON with exactly this shape: {\"clue\":\"your clue\"}\n"
        )

    strategy_prompt = strategy["prompt"]
    return (
        f"{strategy_prompt}"
        f"Word: {secret_word}\n"
        f"{previous_clue_block}"
        "Output requirements:\n"
        "- Do not use the word itself.\n"
        "- Return JSON with exactly this shape: {\"clue\":\"your phrase\"}\n"
    )


def clean_clue_response(
    response_text: str,
    secret_word: str | None = None,
    fallback_hint: str | None = None,
) -> str:
    clue = _first_output_line(response_text)
    clue = re.sub(r"^(clue|answer|output)\s*:\s*", "", clue, flags=re.IGNORECASE)
    clue = clue.strip().strip("\"'` .,!?;:")

    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", clue)
    if len(words) > 5:
        clue = " ".join(words[:5])
    elif words:
        clue = " ".join(words)
    else:
        clue = ""

    if clue.lower() in {"2 to 5 word clue", "short clue", "your clue"}:
        clue = _fallback_clue(secret_word, fallback_hint)

    if secret_word and _contains_word(clue, secret_word):
        clue = _fallback_clue(secret_word, fallback_hint)

    if not clue:
        clue = _fallback_clue(secret_word, fallback_hint)

    return clue


def _first_output_line(response_text: str) -> str:
    for line in response_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


def _fallback_clue(secret_word: str | None, fallback_hint: str | None) -> str:
    source = fallback_hint or "related idea"
    forbidden_words = set(re.findall(r"[A-Za-z0-9]+", secret_word or "", flags=re.IGNORECASE))
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", source)
        if word.lower() not in {"or", "and", "the", "a", "an"}
        and word.lower() not in {forbidden.lower() for forbidden in forbidden_words}
    ]

    if len(words) >= 2:
        return " ".join(words[:5])

    if words:
        return f"{words[0]} clue"

    return "related clue"
