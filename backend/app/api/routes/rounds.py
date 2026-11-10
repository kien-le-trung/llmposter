from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_app_settings
from app.db.session import get_db
from app.services.game import engine, schemas
from app.services.voting.schemas import VoteResponse

router = APIRouter(prefix="/rounds", tags=["rounds"])


def get_inference_http_client(request: Request):
    return request.app.state.inference_http_client


def _http_error(exc: engine.GameError) -> HTTPException:
    status = {"not_found": 404, "invalid_input": 422, "conflict": 409, "configuration": 500, "unavailable": 503}
    return HTTPException(status_code=status[exc.kind], detail=str(exc))


@router.post("", response_model=schemas.RoundResponse, status_code=201)
async def create_round(
    background_tasks: BackgroundTasks,
    payload: schemas.CreateRoundRequest | None = None,
    settings: Settings = Depends(get_app_settings),
    http_client=Depends(get_inference_http_client),
    db: Session = Depends(get_db),
) -> schemas.RoundResponse:
    try:
        return await engine.create_round(payload, settings, db, http_client=http_client, schedule=background_tasks.add_task)
    except engine.GameError as exc:
        raise _http_error(exc) from exc


@router.get("/{round_id}", response_model=schemas.RoundResponse)
async def get_round(round_id: str) -> schemas.RoundResponse:
    try:
        return await engine.get_round(round_id)
    except engine.GameError as exc:
        raise _http_error(exc) from exc


@router.post("/{round_id}/clue", response_model=schemas.RoundResponse)
async def submit_clue(
    round_id: str,
    payload: schemas.SubmitClueRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    http_client=Depends(get_inference_http_client),
) -> schemas.RoundResponse:
    try:
        return await engine.submit_clue(
            round_id, payload, db, settings, http_client=http_client, schedule=background_tasks.add_task,
        )
    except engine.GameError as exc:
        raise _http_error(exc) from exc


@router.post("/{round_id}/vote", response_model=VoteResponse)
async def vote(
    round_id: str,
    payload: schemas.VoteRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    http_client=Depends(get_inference_http_client),
) -> VoteResponse:
    try:
        return await engine.vote(round_id, payload, db, settings, http_client=http_client)
    except engine.GameError as exc:
        raise _http_error(exc) from exc



@router.post("/{round_id}/discussion", response_model=schemas.RoundResponse)
async def submit_discussion_comment(
    round_id: str,
    payload: schemas.SubmitDiscussionComment,
    db: Session = Depends(get_db),
) -> schemas.RoundResponse:
    try:
        return await engine.submit_discussion_comment(round_id, payload, db)
    except engine.GameError as exc:
        raise _http_error(exc) from exc


@router.post("/{round_id}/word-feedback", response_model=schemas.WordFeedbackResponse)
async def submit_word_feedback(
    round_id: str,
    payload: schemas.SubmitWordFeedback,
    db: Session = Depends(get_db),
) -> schemas.WordFeedbackResponse:
    try:
        return await engine.submit_word_feedback(round_id, payload, db)
    except engine.GameError as exc:
        raise _http_error(exc) from exc
