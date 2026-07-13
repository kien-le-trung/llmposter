from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class AgentConfigModel(Base):
    __tablename__ = "agent_configs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False, default="candidate")
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, nullable=False, default=0.7)
    top_p: Mapped[float] = mapped_column(Float, nullable=False, default=0.9)
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=24)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ClueModel(Base):
    __tablename__ = "clues"
    __table_args__ = (UniqueConstraint("round_id", "player_id", "sequence"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    round_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    player_id: Mapped[str] = mapped_column(String(64), nullable=False)
    player_name: Mapped[str] = mapped_column(String(120), nullable=False)
    clue: Mapped[str] = mapped_column(Text, nullable=False)
    inference_mode: Mapped[str | None] = mapped_column(String(64))
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class VoteModel(Base):
    __tablename__ = "votes"
    __table_args__ = (UniqueConstraint("round_id", "voter_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    round_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    voter_id: Mapped[str] = mapped_column(String(64), nullable=False)
    voter_name: Mapped[str] = mapped_column(String(120), nullable=False)
    voted_for_id: Mapped[str | None] = mapped_column(String(64))
    voted_for_name: Mapped[str | None] = mapped_column(String(120))
    raw_vote: Mapped[str | None] = mapped_column(Text)
    inference_mode: Mapped[str | None] = mapped_column(String(64))
    imposter_won: Mapped[bool | None] = mapped_column(Boolean)
    round_winner: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
