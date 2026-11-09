import asyncio
from dataclasses import FrozenInstanceError

import pytest

from app.services.agents import clue_agent
from app.services.agents.inference import InferenceResult, InferenceServiceError
from app.services.agents.runtime_agents import list_static_agent_configs


@pytest.mark.parametrize("imposter", [True, False])
def test_graph_isolates_knowledge_and_selects_strategy(monkeypatch, imposter):
    captured = []
    counts = []
    context = clue_agent.ClueContext(
        None if imposter else "satellite", "orbit", (("Other", "earlier clue"),), "meta",
    )

    def strategy(technique, count):
        counts.append((technique, count))
        return {"name": "test", "prompt": "Use supplied knowledge. "}

    monkeypatch.setattr(clue_agent, "assign_imposter_clue_strategy", strategy)

    class Client:
        async def generate_structured(self, request, response_model):
            captured.append(request.prompt)
            return response_model(clue="one two three four five six"), InferenceResult(
                text="", inference_mode="test",
            )

    result = asyncio.run(clue_agent.run_clue(context, list_static_agent_configs()[0], Client()))
    assert result == {"clue": "one two three four five", "inference_mode": "test"}
    assert ("satellite" in captured[0]) is not imposter
    assert ("earlier clue" in captured[0]) is imposter
    assert ("Hint: orbit" in captured[0]) is imposter
    assert counts == ([("meta", 1)] if imposter else [])
    assert context.previous_clues == (("Other", "earlier clue"),)
    with pytest.raises(FrozenInstanceError):
        context.secret_word = "changed"


def test_graph_propagates_failure_without_mutating_context():
    context = clue_agent.ClueContext("satellite", None, (), "few_shot")

    class Client:
        async def generate_structured(self, *args):
            raise InferenceServiceError("unavailable")

    with pytest.raises(InferenceServiceError):
        asyncio.run(clue_agent.run_clue(context, list_static_agent_configs()[0], Client()))
    assert context.secret_word == "satellite"


def test_secret_word_and_empty_clues_are_normalized():
    assert clue_agent.clean_clue_response("satellite", "satellite") == "related idea"
    assert clue_agent.clean_clue_response("", None, "orbit") == "orbit clue"
