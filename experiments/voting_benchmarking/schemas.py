from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    secret_word: str = Field(min_length=1)
    imposter_hint: str = Field(min_length=1)
    human_clue: str | None = Field(default=None, min_length=1)


class BenchmarkConfig(BaseModel):
    backend_url: str
    technique: str
    repetitions: int = Field(ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    compute_semantic_features: bool = True
    require_semantic_features: bool = False
    show_progress: bool = False
    progress_every: int = Field(default=1, ge=1)


class PlayerRecord(BaseModel):
    player_id: str
    player_name: str
    player_kind: str
    position: int


class RoundArtifactRecord(BaseModel):
    run_id: str
    technique: str
    round_id: str | None
    secret_word: str
    imposter_hint: str
    human_clue: str | None = None
    status: str
    success: bool
    latency_ms: float
    playing_order: list[PlayerRecord]
    imposter_player_id: str | None = None
    imposter_player_name: str | None = None
    imposter_kind: str | None = None
    num_players: int = 0
    error: str | None = None


class ClueRecord(BaseModel):
    run_id: str
    technique: str
    round_id: str
    secret_word: str
    imposter_hint: str
    player_id: str
    player_name: str
    player_kind: str
    role: str | None
    position: int
    clue: str
    inference_mode: str
    previous_clue_count: int
    is_first: bool
    is_last: bool
    is_before_human: bool
    is_after_human: bool


class AgentVoteRecord(BaseModel):
    voter_agent_id: str
    voter_agent_name: str
    voted_for_player_id: str | None
    voted_for_player_name: str
    voted_for_is_imposter: bool | None
    inference_mode: str


class VoteCountRecord(BaseModel):
    player_id: str
    player_name: str
    votes: int
    is_imposter: bool | None


class VoteRecord(BaseModel):
    run_id: str
    technique: str
    round_id: str
    voting_algorithm: str
    human_vote_strategy: str
    human_voted_player_id: str | None
    group_voted_player_id: str | None
    group_voted_player_name: str | None
    group_voted_is_imposter: bool | None
    imposter_won: bool
    round_winner: str
    agent_votes: list[AgentVoteRecord]
    vote_counts: list[VoteCountRecord]


class SemanticFeatureRecord(BaseModel):
    run_id: str
    technique: str
    round_id: str
    player_id: str
    player_name: str
    role: str | None
    embedding_model: str
    embedding_inference_mode: str
    clue_to_secret_similarity: float
    clue_to_hint_similarity: float
    clue_to_non_imposter_centroid_similarity: float | None
    non_imposter_pairwise_similarity: float | None
    imposter_outlier_score: float | None
    separability_margin: float | None
    hint_to_secret_similarity: float


class BenchmarkSummary(BaseModel):
    technique: str
    completed_rounds: int
    failed_rounds: int
    agent_imposter_rounds: int
    human_imposter_rounds: int
    average_latency_ms: float
    round_success_rate: float
    agent_vote_detection_rate: float
    agent_only_group_detection_rate: float
    group_detection_rate: float
    random_chance_detection_rate: float
    detection_lift_over_random: float
    mean_hint_to_secret_similarity: float | None
    mean_non_imposter_clue_to_secret_similarity: float | None
    mean_imposter_clue_to_secret_similarity: float | None
    mean_non_imposter_pairwise_similarity: float | None
    mean_imposter_outlier_score: float | None
    mean_separability_margin: float | None
    detection_rate_by_imposter_position: dict[str, float]
