from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import Settings, get_app_settings
from app.db.session import get_db
from app.services.agents.runtime_agents import list_runtime_agent_configs

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentResponse(BaseModel):
    id: str
    name: str
    role: str
    version: str


@router.get("", response_model=list[AgentResponse])
def list_agents(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> list[AgentResponse]:
    agents = list_runtime_agent_configs(db, settings)
    return [
        AgentResponse(id=agent.id, name=agent.name, role=agent.role, version=agent.version)
        for agent in agents
    ]
