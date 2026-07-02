from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AgentConfigModel
from app.db.session import SessionLocal, create_db_and_tables
from app.services.agents import list_agent_configs


def seed_agent_configs(db: Session) -> None:
    for agent in list_agent_configs():
        existing = db.scalar(
            select(AgentConfigModel).where(AgentConfigModel.id == agent.id)
        )
        if existing is not None:
            continue

        db.add(
            AgentConfigModel(
                id=agent.id,
                name=agent.name,
                role=agent.role,
                system_prompt=agent.system_prompt,
                temperature=agent.temperature,
                top_p=agent.top_p,
                max_tokens=agent.max_tokens,
                version=agent.version,
            )
        )

    db.commit()


def init_database() -> None:
    create_db_and_tables()
    with SessionLocal() as db:
        seed_agent_configs(db)
