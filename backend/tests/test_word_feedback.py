from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.models import WordFeedbackModel
from app.db.session import Base, get_db
from app.main import create_app
from app.services.game.state import ROUNDS, RoundState


@pytest.fixture
def feedback_client():
    database = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(database)
    state = RoundState(
        id="feedback-round", secret_word="satellite", imposter_hint="orbit",
        imposter_player_id="agent_a", status="discussion", playing_order=["human", "agent_a"],
        player_names_by_id={"human": "You", "agent_a": "Maya"}, prompt_technique="few_shot",
        current_player_index=2, turns=[], created_at=datetime.now(UTC),
    )
    ROUNDS[state.id] = state
    app = create_app(Settings(_env_file=None, inference_mode="fake"))
    with Session(database) as db:
        app.dependency_overrides[get_db] = lambda: db
        with TestClient(app) as client:
            yield client, db, state, app
    ROUNDS.pop(state.id, None)
    database.dispose()


@pytest.mark.parametrize("answer", ["yes", "no"])
@pytest.mark.parametrize("imposter", [False, True])
def test_feedback_persists_server_word_and_role(feedback_client, answer, imposter):
    client, db, state, _ = feedback_client
    if imposter:
        state.imposter_player_id = "human"
    response = client.post(f"/rounds/{state.id}/word-feedback", json={"answer": answer})
    assert response.status_code == 200
    assert response.json() == {"round_id": state.id, "answer": answer}
    db.expire_all()
    saved = db.get(WordFeedbackModel, state.id)
    assert saved.answer == answer
    assert saved.secret_word == "satellite"
    assert saved.given_word == ("orbit" if imposter else "satellite")
    assert saved.user_role == ("imposter" if imposter else "player")
    assert saved.created_at is not None
    assert saved.updated_at is not None


def test_retries_and_changed_answers_update_one_row(feedback_client):
    client, db, state, _ = feedback_client
    url = f"/rounds/{state.id}/word-feedback"
    for answer in ["yes", "yes", "no"]:
        assert client.post(url, json={"answer": answer}).status_code == 200
    saved = db.scalars(select(WordFeedbackModel)).all()
    assert len(saved) == 1
    assert saved[0].answer == "no"


@pytest.mark.parametrize("status", ["ready_to_vote", "complete"])
def test_feedback_can_retry_after_discussion_ends(feedback_client, status):
    client, _, state, _ = feedback_client
    state.status = status
    assert client.post(f"/rounds/{state.id}/word-feedback", json={"answer": "yes"}).status_code == 200


@pytest.mark.parametrize("payload", [{}, {"answer": "maybe"}, {"answer": True}])
def test_invalid_feedback_is_rejected(feedback_client, payload):
    client, db, state, _ = feedback_client
    assert client.post(f"/rounds/{state.id}/word-feedback", json=payload).status_code == 422
    assert db.scalars(select(WordFeedbackModel)).all() == []


def test_missing_round_and_ineligible_rounds_are_rejected(feedback_client):
    client, db, state, _ = feedback_client
    assert client.post("/rounds/missing/word-feedback", json={"answer": "yes"}).status_code == 404
    state.status = "awaiting_human_clue"
    assert client.post(f"/rounds/{state.id}/word-feedback", json={"answer": "yes"}).status_code == 409
    state.status = "discussion"
    state.include_human = False
    assert client.post(f"/rounds/{state.id}/word-feedback", json={"answer": "yes"}).status_code == 409
    assert db.scalars(select(WordFeedbackModel)).all() == []


def test_database_failure_rolls_back_and_allows_retry(feedback_client):
    client, db, state, app = feedback_client
    failed_db = Mock(spec=Session)
    failed_db.commit.side_effect = SQLAlchemyError("private connection information")
    failed_db.get.return_value = None
    app.dependency_overrides[get_db] = lambda: failed_db
    url = f"/rounds/{state.id}/word-feedback"
    response = client.post(url, json={"answer": "yes"})
    assert response.status_code == 503
    assert "private" not in response.text
    failed_db.rollback.assert_called_once()
    app.dependency_overrides[get_db] = lambda: db
    assert client.post(url, json={"answer": "yes"}).status_code == 200
