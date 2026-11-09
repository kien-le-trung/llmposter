from pathlib import Path

from alembic import command
from alembic.config import Config

from app.db.session import engine


def upgrade_database() -> None:
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


if __name__ == "__main__":
    upgrade_database()
