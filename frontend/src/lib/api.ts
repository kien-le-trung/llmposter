const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type HealthResponse = {
  status: string;
  app: string;
  environment: string;
};

export type Agent = {
  id: string;
  name: string;
  role: string;
  version: string;
};

export type GenerateResponse = {
  agent_id: string;
  text: string;
  inference_mode: string;
};

export type AgentTurnResponse = {
  agent_id: string;
  agent_name: string;
  agent_response: string;
  inference_mode: string;
};

export type Turn = {
  id: string;
  sequence: number;
  user_prompt: string;
  responses: AgentTurnResponse[];
  created_at: string;
};

export type Round = {
  id: string;
  visible_word: string | null;
  user_role: "player" | "imposter";
  status: string;
  turns: Turn[];
  created_at: string;
};

export type VoteResult = {
  voted_agent_id: string;
  voted_agent_name: string;
  correct: boolean;
  imposter_was: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function getHealth() {
  return request<HealthResponse>("/health");
}

export function getAgents() {
  return request<Agent[]>("/agents");
}

export function generateAgentResponse(prompt: string, agentId = "host") {
  return request<GenerateResponse>("/agents/generate", {
    method: "POST",
    body: JSON.stringify({ prompt, agent_id: agentId }),
  });
}

export function createRound(secretWord: string) {
  return request<Round>("/rounds", {
    method: "POST",
    body: JSON.stringify({ secret_word: secretWord }),
  });
}

export function getRound(roundId: string) {
  return request<Round>(`/rounds/${roundId}`);
}

export function voteRound(roundId: string, agentId: string) {
  return request<VoteResult>(`/rounds/${roundId}/vote`, {
    method: "POST",
    body: JSON.stringify({ agent_id: agentId }),
  });
}
