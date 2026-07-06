from app import prompts
from app.services.inference import AgentConfig

build_clue_system_prompt = prompts.build_clue_system_prompt
build_clue_user_prompt = prompts.build_clue_user_prompt
build_batched_clue_system_prompt = prompts.build_batched_clue_system_prompt
build_batched_clue_user_prompt = prompts.build_batched_clue_user_prompt
build_instruction_batched_clue_system_prompt = (
    prompts.build_instruction_batched_clue_system_prompt
)
build_instruction_batched_clue_user_prompt = prompts.build_instruction_batched_clue_user_prompt
build_instruction_clue_system_prompt = prompts.build_instruction_clue_system_prompt
build_instruction_clue_user_prompt = prompts.build_instruction_clue_user_prompt
IMPOSTER_CLUE_STRATEGIES = prompts.IMPOSTER_CLUE_STRATEGIES
NON_IMPOSTER_CLUE_STRATEGIES = prompts.NON_IMPOSTER_CLUE_STRATEGIES
PROMPT_TECHNIQUES = prompts.PROMPT_TECHNIQUES
load_imposter_clue_strategies = prompts.load_imposter_clue_strategies
load_non_imposter_clue_strategies = prompts.load_non_imposter_clue_strategies
normalize_prompt_technique = prompts.normalize_prompt_technique
build_vote_system_prompt = prompts.build_vote_system_prompt
build_vote_user_prompt = prompts.build_vote_user_prompt
clean_batched_clue_response = prompts.clean_batched_clue_response
clean_clue_response = prompts.clean_clue_response
clean_vote_response = prompts.clean_vote_response

AGENTS: dict[str, AgentConfig] = {
    "agent_a": AgentConfig(
        id="agent_a",
        name="Agent A",
        role="candidate",
        system_prompt="You are a player in a word-based imposter game.",
        temperature=0.3,
        top_p=0.8,
        max_tokens=24,
        version="prompt-v3",
    ),
    "agent_b": AgentConfig(
        id="agent_b",
        name="Agent B",
        role="candidate",
        system_prompt="You are a player in a word-based imposter game.",
        temperature=0.3,
        top_p=0.8,
        max_tokens=24,
        version="prompt-v3",
    ),
    "agent_c": AgentConfig(
        id="agent_c",
        name="Agent C",
        role="candidate",
        system_prompt="You are a player in a word-based imposter game.",
        temperature=0.3,
        top_p=0.8,
        max_tokens=24,
        version="prompt-v3",
    ),
    "agent_d": AgentConfig(
        id="agent_d",
        name="Agent D",
        role="candidate",
        system_prompt="You are a player in a word-based imposter game.",
        temperature=0.3,
        top_p=0.8,
        max_tokens=24,
        version="prompt-v3",
    ),
}


def list_agent_configs() -> list[AgentConfig]:
    return list(AGENTS.values())


def get_agent(agent_id: str) -> AgentConfig | None:
    return AGENTS.get(agent_id)
