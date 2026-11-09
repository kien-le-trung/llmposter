"""Frozen schema: create missing tables or adopt compatible existing tables."""

from alembic import op
import sqlalchemy as sa

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("agent_configs"):
        columns = {c["name"] for c in inspector.get_columns("agent_configs")}
        if not set(
            [
                "id",
                "name",
                "role",
                "system_prompt",
                "temperature",
                "top_p",
                "max_tokens",
                "version",
                "is_active",
                "created_at",
                "updated_at",
            ]
        ).issubset(columns):
            raise RuntimeError(
                "Incompatible existing agent_configs columns; reconcile the schema before upgrading."
            )
    else:
        op.create_table(
            "agent_configs",
            sa.Column("id", sa.String(length=64), nullable=False, primary_key=True),
            sa.Column("name", sa.String(length=120), nullable=False),
            sa.Column("role", sa.String(length=64), nullable=False),
            sa.Column("system_prompt", sa.Text(), nullable=False),
            sa.Column("temperature", sa.Float(), nullable=False),
            sa.Column("top_p", sa.Float(), nullable=False),
            sa.Column("max_tokens", sa.Integer(), nullable=False),
            sa.Column("version", sa.String(length=64), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
        )
    if inspector.has_table("clues"):
        columns = {c["name"] for c in inspector.get_columns("clues")}
        if not set(
            [
                "id",
                "round_id",
                "player_id",
                "player_name",
                "clue",
                "inference_mode",
                "sequence",
                "created_at",
            ]
        ).issubset(columns):
            raise RuntimeError(
                "Incompatible existing clues columns; reconcile the schema before upgrading."
            )
        if inspector.get_foreign_keys("clues"):
            raise RuntimeError(
                "Legacy clues foreign keys detected; reconcile the round-linked schema before upgrading. No data was deleted."
            )
    else:
        op.create_table(
            "clues",
            sa.Column("id", sa.String(length=36), nullable=False, primary_key=True),
            sa.Column("round_id", sa.String(length=36), nullable=False),
            sa.Column("player_id", sa.String(length=64), nullable=False),
            sa.Column("player_name", sa.String(length=120), nullable=False),
            sa.Column("clue", sa.Text(), nullable=False),
            sa.Column("inference_mode", sa.String(length=64), nullable=True),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.UniqueConstraint("round_id", "player_id", "sequence"),
        )
        op.create_index("ix_clues_round_id", "clues", ["round_id"])
    if inspector.has_table("votes"):
        columns = {c["name"] for c in inspector.get_columns("votes")}
        if not set(
            [
                "id",
                "round_id",
                "voter_id",
                "voter_name",
                "voted_for_id",
                "voted_for_name",
                "raw_vote",
                "inference_mode",
                "imposter_won",
                "round_winner",
                "created_at",
            ]
        ).issubset(columns):
            raise RuntimeError(
                "Incompatible existing votes columns; reconcile the schema before upgrading."
            )
        if inspector.get_foreign_keys("votes"):
            raise RuntimeError(
                "Legacy votes foreign keys detected; reconcile the round-linked schema before upgrading. No data was deleted."
            )
    else:
        op.create_table(
            "votes",
            sa.Column("id", sa.String(length=36), nullable=False, primary_key=True),
            sa.Column("round_id", sa.String(length=36), nullable=False),
            sa.Column("voter_id", sa.String(length=64), nullable=False),
            sa.Column("voter_name", sa.String(length=120), nullable=False),
            sa.Column("voted_for_id", sa.String(length=64), nullable=True),
            sa.Column("voted_for_name", sa.String(length=120), nullable=True),
            sa.Column("raw_vote", sa.Text(), nullable=True),
            sa.Column("inference_mode", sa.String(length=64), nullable=True),
            sa.Column("imposter_won", sa.Boolean(), nullable=True),
            sa.Column("round_winner", sa.String(length=16), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.UniqueConstraint("round_id", "voter_id"),
        )
        op.create_index("ix_votes_round_id", "votes", ["round_id"])


def downgrade():
    raise RuntimeError(
        "Automatic downgrade is disabled: adopted tables may contain pre-existing data."
    )
