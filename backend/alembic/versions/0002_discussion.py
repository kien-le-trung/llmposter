"""Frozen schema: create missing tables or adopt compatible existing tables."""

from alembic import op
import sqlalchemy as sa

revision = "0002_discussion"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("discussion_messages"):
        columns = {c["name"] for c in inspector.get_columns("discussion_messages")}
        if not set(
            [
                "id",
                "round_id",
                "player_id",
                "player_name",
                "text",
                "sequence",
                "inference_mode",
                "created_at",
            ]
        ).issubset(columns):
            raise RuntimeError(
                "Incompatible existing discussion_messages columns; reconcile the schema before upgrading."
            )
    else:
        op.create_table(
            "discussion_messages",
            sa.Column("id", sa.String(length=36), nullable=False, primary_key=True),
            sa.Column("round_id", sa.String(length=36), nullable=False),
            sa.Column("player_id", sa.String(length=64), nullable=False),
            sa.Column("player_name", sa.String(length=120), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("inference_mode", sa.String(length=64), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("round_id", "sequence"),
        )
        op.create_index("ix_discussion_messages_round_id", "discussion_messages", ["round_id"])


def downgrade():
    raise RuntimeError(
        "Automatic downgrade is disabled: adopted tables may contain pre-existing data."
    )
