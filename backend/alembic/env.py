from alembic import context

from app.db import models  # noqa: F401
from app.db.session import Base, engine

config = context.config


def run_migrations(connection):
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("These adoption migrations require a live database connection.")
elif config.attributes.get("connection") is not None:
    run_migrations(config.attributes["connection"])
else:
    with engine.connect() as connection:
        run_migrations(connection)
