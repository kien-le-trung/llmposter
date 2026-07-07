from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AgentConfigModel
from app.services.agents.inference import AgentConfig


def model_to_agent_config(agent: AgentConfigModel) -> AgentConfig:
    return AgentConfig(
        id=agent.id,
        name=agent.name,
        role=agent.role,
        system_prompt=agent.system_prompt,
        temperature=agent.temperature,
        top_p=agent.top_p,
        max_tokens=agent.max_tokens,
        version=agent.version,
    )


def list_active_agent_configs(db: Session) -> list[AgentConfig]:
    results = db.scalars(
        select(AgentConfigModel)
        .where(AgentConfigModel.is_active.is_(True))
        .order_by(AgentConfigModel.id)
    )
    return [model_to_agent_config(agent) for agent in results]


def get_active_agent_config(db: Session, agent_id: str) -> AgentConfig | None:
    agent = db.scalar(
        select(AgentConfigModel).where(
            AgentConfigModel.id == agent_id,
            AgentConfigModel.is_active.is_(True),
        )
    )
    if agent is None:
        return None

    return model_to_agent_config(agent)
