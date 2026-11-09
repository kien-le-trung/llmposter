from datetime import datetime

from pydantic import BaseModel, Field

from app.services.agents import discussion as chat
from app.services.game.schemas import AgentTurnResponse, TurnResponse

HUMAN_PLAYER_ID = "human"


class RoundState(BaseModel):
    id: str
    secret_word: str
    imposter_hint: str
    imposter_player_id: str
    status: str
    playing_order: list[str]
    player_names_by_id: dict[str, str]
    prompt_technique: str
    current_player_index: int
    turns: list[TurnResponse]
    created_at: datetime
    include_human: bool = True
    discussion: chat.DiscussionState = Field(default_factory=chat.DiscussionState)
    discussion_duration_seconds: int = 90
    human_clue: str | None = None
    voted_agent_id: str | None = None
    completed_clues: dict[str, AgentTurnResponse] = Field(default_factory=dict)


# Process-wide round state; not persisted.
ROUNDS: dict[str, RoundState] = {}
ACTIVE_GENERATION_ROUNDS: set[str] = set()
