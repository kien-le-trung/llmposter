from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.agent_config_store import get_active_agent_config, list_active_agent_configs
from app.services.agents import get_agent, list_agent_configs
from app.services.inference import AgentConfig


def list_runtime_agent_configs(db: Session) -> list[AgentConfig]:
    if settings.agent_config_source != "database":
        return list_agent_configs()

    try:
        agents = list_active_agent_configs(db)
    except SQLAlchemyError:
        return list_agent_configs()

    return agents or list_agent_configs()


def get_runtime_agent_config(db: Session, agent_id: str) -> AgentConfig | None:
    if settings.agent_config_source != "database":
        return get_agent(agent_id)

    try:
        agent = get_active_agent_config(db, agent_id)
    except SQLAlchemyError:
        return get_agent(agent_id)

    return agent or get_agent(agent_id)
