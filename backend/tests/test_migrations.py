from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import (
    Column,
    ForeignKey,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
    select,
    text,
)

from app.db.models import AgentConfigModel, ClueModel, DiscussionMessageModel, VoteModel
from app.db.session import Base


def config_for(connection):
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    config.attributes["connection"] = connection
    return config


@pytest.mark.parametrize("existing", ["empty", "baseline", "current"])
def test_upgrade_preserves_data_and_matches_models(existing):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        if existing != "empty":
            tables = [AgentConfigModel.__table__, ClueModel.__table__, VoteModel.__table__]
            if existing == "current":
                tables.append(DiscussionMessageModel.__table__)
            Base.metadata.create_all(connection, tables=tables)
            connection.execute(
                ClueModel.__table__.insert().values(
                    id="old-clue",
                    round_id="old-round",
                    player_id="human",
                    player_name="You",
                    clue="preserved",
                    sequence=1,
                )
            )
        config = config_for(connection)
        command.upgrade(config, "head")
        command.upgrade(config, "head")
        assert (
            connection.scalar(text("SELECT version_num FROM alembic_version")) == "0002_discussion"
        )
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
        if existing != "empty":
            assert connection.scalar(select(ClueModel.clue)) == "preserved"
        connection.execute(
            DiscussionMessageModel.__table__.insert().values(
                id="message",
                round_id="new-round",
                player_id="human",
                player_name="You",
                text="hello",
                sequence=1,
                inference_mode="human",
                created_at=datetime.now(UTC),
            )
        )


def test_upgrade_rejects_incompatible_existing_schema():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE clues (id VARCHAR(36) PRIMARY KEY)"))
        connection.execute(text("INSERT INTO clues VALUES ('keep-me')"))
        with pytest.raises(RuntimeError, match="Incompatible existing clues"):
            command.upgrade(config_for(connection), "head")
        assert connection.scalar(text("SELECT id FROM clues")) == "keep-me"
        assert "discussion_messages" not in inspect(connection).get_table_names()


def test_upgrade_rejects_legacy_round_foreign_keys():
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    Table("rounds", metadata, Column("id", String(36), primary_key=True))
    clues = ClueModel.__table__.to_metadata(metadata)
    clues.c.round_id.append_foreign_key(ForeignKey("rounds.id"))
    with engine.begin() as connection:
        metadata.create_all(connection)
        with pytest.raises(RuntimeError, match="Legacy clues foreign keys"):
            command.upgrade(config_for(connection), "head")
        assert "rounds" in inspect(connection).get_table_names()
