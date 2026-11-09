from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.services.agents import discussion as chat


class AgentTurnResponse(BaseModel):
    agent_id: str
    agent_name: str
    agent_response: str
    inference_mode: str


class TurnResponse(BaseModel):
    id: str
    sequence: int
    user_prompt: str
    responses: list[AgentTurnResponse]
    created_at: datetime


class PlayerResponse(BaseModel):
    id: str
    name: str
    kind: str


class RoundResponse(BaseModel):
    id: str
    visible_word: str | None
    imposter_hint: str | None
    user_role: str
    prompt_technique: str
    status: str
    discussion: chat.DiscussionState
    playing_order: list[PlayerResponse]
    current_player_id: str | None
    current_player_name: str | None
    turns: list[TurnResponse]
    created_at: datetime


class CreateRoundRequest(BaseModel):
    human_name: str | None = Field(default=None, min_length=1, max_length=40)
    secret_word: str | None = Field(default=None, min_length=1, max_length=80)
    imposter_hint: str | None = Field(default=None, min_length=1, max_length=160)
    prompt_technique: str | None = Field(default=None, min_length=1, max_length=80)
    include_human: bool = True

    @field_validator("human_name", mode="before")
    @classmethod
    def trim_human_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class VoteRequest(BaseModel):
    agent_id: str | None = None


class SubmitClueRequest(BaseModel):
    clue: str = Field(min_length=1, max_length=200)


class SubmitDiscussionComment(BaseModel):
    text: str = Field(min_length=1, max_length=500)
