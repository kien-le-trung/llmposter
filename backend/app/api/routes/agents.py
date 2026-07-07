from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.agents.runtime_agents import list_runtime_agent_configs

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentResponse(BaseModel):
    id: str
    name: str
    role: str
    version: str


@router.get("", response_model=list[AgentResponse])
def list_agents(db: Session = Depends(get_db)) -> list[AgentResponse]:
    agents = list_runtime_agent_configs(db)
    return [
        AgentResponse(id=agent.id, name=agent.name, role=agent.role, version=agent.version)
        for agent in agents
    ]
