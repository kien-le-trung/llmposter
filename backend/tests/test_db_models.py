import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import AgentConfigModel, ClueModel, VoteModel
from app.db.session import Base
from app.core.config import Settings
from app.services.agents.runtime_agents import get_runtime_agent_config, list_runtime_agent_configs


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
    settings = Settings(_env_file=None)
    settings.agent_config_source = "database"
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

        agents = list_runtime_agent_configs(db, settings)
        agent = get_runtime_agent_config(db, "agent_db", settings)

    assert [agent.id for agent in agents] == ["agent_db"]
    assert agent is not None
    assert agent.name == "Database Agent"
    assert agent.temperature == 0.2


def test_clues_and_votes_can_persist_without_round_parent() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    with Session(engine) as db:
        db.add_all([
            ClueModel(id="clue_h", round_id="round_test", player_id="human",
                      player_name="You", clue="circles earth", sequence=1),
            ClueModel(id="clue_a", round_id="round_test", player_id="agent_a",
                      player_name="Agent A", clue="space object", sequence=1,
                      inference_mode="fake"),
            VoteModel(id="vote_h", round_id="round_test", voter_id="human",
                      voter_name="You", voted_for_id="agent_a", voted_for_name="Agent A"),
            VoteModel(id="vote_a", round_id="round_test", voter_id="agent_a",
                      voter_name="Agent A", voted_for_id="human", voted_for_name="You",
                      raw_vote="You", inference_mode="ml_voting"),
            VoteModel(id="vote_group", round_id="round_test", voter_id="group",
                      voter_name="Group", voted_for_id="agent_a", voted_for_name="Agent A",
                      inference_mode="tally", imposter_won=False, round_winner="players"),
        ])
        db.commit()
        clues = db.scalars(select(ClueModel).where(ClueModel.round_id == "round_test")).all()
        votes = db.scalars(select(VoteModel).where(VoteModel.round_id == "round_test")).all()
        assert {clue.clue for clue in clues} == {"circles earth", "space object"}
        assert {vote.voter_id for vote in votes} == {"human", "agent_a", "group"}
        group_vote = next(vote for vote in votes if vote.voter_id == "group")
        assert group_vote.imposter_won is False
        assert group_vote.round_winner == "players"


@pytest.mark.parametrize("child_type", ["clue", "vote"])
def test_duplicate_round_player_records_are_rejected(child_type: str) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    with Session(engine) as db:
        if child_type == "clue":
            records = [
                ClueModel(id="one", round_id="round_test", player_id="human",
                          player_name="You", clue="one", sequence=1),
                ClueModel(id="two", round_id="round_test", player_id="human",
                          player_name="You", clue="two", sequence=1),
            ]
        else:
            records = [
                VoteModel(id="one", round_id="round_test", voter_id="human", voter_name="You"),
                VoteModel(id="two", round_id="round_test", voter_id="human", voter_name="You"),
            ]
        db.add_all(records)
        with pytest.raises(IntegrityError):
            db.commit()


def test_round_table_and_child_foreign_keys_are_absent() -> None:
    assert "rounds" not in Base.metadata.tables
    assert not ClueModel.__table__.foreign_keys
    assert not VoteModel.__table__.foreign_keys
