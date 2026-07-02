from app.services.inference import AgentConfig

AGENTS: dict[str, AgentConfig] = {
    "agent_a": AgentConfig(
        id="agent_a",
        name="Agent A",
        role="candidate",
        system_prompt="You are a player in a word-based imposter game.",
        temperature=0.7,
        top_p=0.9,
        max_tokens=24,
        version="prompt-v2",
    ),
    "agent_b": AgentConfig(
        id="agent_b",
        name="Agent B",
        role="candidate",
        system_prompt="You are a player in a word-based imposter game.",
        temperature=0.7,
        top_p=0.9,
        max_tokens=24,
        version="prompt-v2",
    ),
    "agent_c": AgentConfig(
        id="agent_c",
        name="Agent C",
        role="candidate",
        system_prompt="You are a player in a word-based imposter game.",
        temperature=0.7,
        top_p=0.9,
        max_tokens=24,
        version="prompt-v2",
    ),
    "agent_d": AgentConfig(
        id="agent_d",
        name="Agent D",
        role="candidate",
        system_prompt="You are a player in a word-based imposter game.",
        temperature=0.7,
        top_p=0.9,
        max_tokens=24,
        version="prompt-v2",
    ),
}


def list_agent_configs() -> list[AgentConfig]:
    return list(AGENTS.values())


def get_agent(agent_id: str) -> AgentConfig | None:
    return AGENTS.get(agent_id)


def build_clue_system_prompt(secret_word: str | None) -> str:
    if secret_word is None:
        return (
            "Reply with a random natural phrase. "
            "Strictly 2-5 words. No explanation."
        )

    return (
        f"Secret word: {secret_word}. "
        f"Describe it without using '{secret_word}'. "
        "Strictly 2-5 words. No explanation."
    )
