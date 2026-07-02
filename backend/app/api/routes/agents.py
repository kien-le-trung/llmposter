from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.agents import get_agent, list_agent_configs
from app.services.inference import (
    InferenceClient,
    InferenceRequest,
    InferenceServiceError,
)

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
def list_agents() -> list[AgentResponse]:
    return [
        AgentResponse(id=agent.id, name=agent.name, role=agent.role, version=agent.version)
        for agent in list_agent_configs()
    ]


@router.post("/generate", response_model=GenerateResponse)
async def generate_response(payload: GenerateRequest) -> GenerateResponse:
    agent = get_agent(payload.agent_id)
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
