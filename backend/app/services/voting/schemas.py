from typing import TypedDict
from pydantic import BaseModel

class VotingFeatureInput(TypedDict):
    round_id: str
    turn_id: str
    candidate_turn_position: int
    candidate_embedding: list[float]
    other_embeddings: list[list[float]]
    previous_embeddings: list[list[float]]

class AgentVoteResponse(BaseModel):
    voter_agent_id: str
    voter_agent_name: str
    voted_for: str
    inference_mode: str


class VoteCountResponse(BaseModel):
    player_id: str
    player_name: str
    votes: int


class VoteResponse(BaseModel):
    voted_agent_id: str | None
    voted_agent_name: str | None
    secret_word: str
    imposter_was: str
    agent_votes: list[AgentVoteResponse]
    vote_counts: list[VoteCountResponse]
    group_voted_player_id: str | None
    group_voted_player_name: str | None
    imposter_won: bool
    round_winner: str