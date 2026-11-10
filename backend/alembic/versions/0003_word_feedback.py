"""Persist one word suitability response per human round."""

from alembic import op
import sqlalchemy as sa

revision = "0003_word_feedback"
down_revision = "0002_discussion"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "word_feedback",
        sa.Column("round_id", sa.String(length=36), primary_key=True),
        sa.Column("secret_word", sa.String(length=80), nullable=False),
        sa.Column("given_word", sa.String(length=160), nullable=False),
        sa.Column("user_role", sa.String(length=16), nullable=False),
        sa.Column("answer", sa.String(length=3), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("answer IN ('yes', 'no')", name="ck_word_feedback_answer"),
    )


def downgrade():
    raise RuntimeError("Automatic downgrade is disabled to preserve collected word feedback.")
