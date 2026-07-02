from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.services.inference import (
    InferenceClient,
    InferenceRequest,
    InferenceServiceError,
)
from app.services.runtime_agents import get_runtime_agent_config, list_runtime_agent_configs

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentResponse(BaseModel):
    id: str
    name: str
    role: str
    version: str


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    agent_id: str = "host"


class GenerateResponse(BaseModel):
    agent_id: str
    text: str
    inference_mode: str


@router.get("", response_model=list[AgentResponse])
def list_agents(db: Session = Depends(get_db)) -> list[AgentResponse]:
    agents = list_runtime_agent_configs(db)
    return [
        AgentResponse(id=agent.id, name=agent.name, role=agent.role, version=agent.version)
        for agent in agents
    ]


@router.post("/generate", response_model=GenerateResponse)
async def generate_response(
    payload: GenerateRequest,
    db: Session = Depends(get_db),
) -> GenerateResponse:
    agent = get_runtime_agent_config(db, payload.agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Unknown agent")

    client = InferenceClient(settings=settings)
    try:
        result = await client.generate(InferenceRequest(prompt=payload.prompt, agent=agent))
    except InferenceServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return GenerateResponse(
        agent_id=agent.id,
        text=result.text,
        inference_mode=result.inference_mode,
    )
