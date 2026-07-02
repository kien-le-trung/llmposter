from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.db.models import AgentConfigModel
from app.db.session import Base
from app.services.runtime_agents import get_runtime_agent_config, list_runtime_agent_configs


def test_agent_config_model_can_persist() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        db.add(
            AgentConfigModel(
                id="agent_test",
                name="Agent Test",
                role="candidate",
                system_prompt="Give a short clue.",
                temperature=0.7,
                top_p=0.9,
                max_tokens=24,
                version="test",
            )
        )
        db.commit()

        agent = db.scalar(
            select(AgentConfigModel).where(AgentConfigModel.id == "agent_test")
        )

    assert agent is not None
    assert agent.name == "Agent Test"
    assert agent.is_active is True


def test_runtime_agent_configs_read_from_database(monkeypatch) -> None:
    from app.services import runtime_agents

    monkeypatch.setattr(runtime_agents.settings, "agent_config_source", "database")
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        db.add(
            AgentConfigModel(
                id="agent_db",
                name="Database Agent",
                role="candidate",
                system_prompt="Use the database config.",
                temperature=0.2,
                top_p=0.8,
                max_tokens=12,
                version="db-test",
            )
        )
        db.commit()

        agents = list_runtime_agent_configs(db)
        agent = get_runtime_agent_config(db, "agent_db")

    assert [agent.id for agent in agents] == ["agent_db"]
    assert agent is not None
    assert agent.name == "Database Agent"
    assert agent.temperature == 0.2
