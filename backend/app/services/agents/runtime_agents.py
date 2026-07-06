import json
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.agents.agent_config_store import get_active_agent_config, list_active_agent_configs
from app.services.agents.inference import AgentConfig

STATIC_AGENT_CONFIGS_PATH = Path(__file__).resolve().parents[2] / "data" / "agents.json"


def list_static_agent_configs() -> list[AgentConfig]:
    return [
        _raw_agent_to_config(agent)
        for agent in json.loads(STATIC_AGENT_CONFIGS_PATH.read_text(encoding="utf-8"))
    ]


def get_static_agent_config(agent_id: str) -> AgentConfig | None:
    return next(
        (agent for agent in list_static_agent_configs() if agent.id == agent_id),
        None,
    )


def list_runtime_agent_configs(db: Session) -> list[AgentConfig]:
    if settings.agent_config_source != "database":
        return list_static_agent_configs()

    try:
        agents = list_active_agent_configs(db)
    except SQLAlchemyError:
        return list_static_agent_configs()

    return agents or list_static_agent_configs()


def get_runtime_agent_config(db: Session, agent_id: str) -> AgentConfig | None:
    if settings.agent_config_source != "database":
        return get_static_agent_config(agent_id)

    try:
        agent = get_active_agent_config(db, agent_id)
    except SQLAlchemyError:
        return get_static_agent_config(agent_id)

    return agent or get_static_agent_config(agent_id)


def _raw_agent_to_config(agent: dict) -> AgentConfig:
    return AgentConfig(
        id=agent["id"],
        name=agent["name"],
        role=agent["role"],
        system_prompt=agent["system_prompt"],
        temperature=agent["temperature"],
        top_p=agent["top_p"],
        max_tokens=agent["max_tokens"],
        version=agent["version"],
    )
