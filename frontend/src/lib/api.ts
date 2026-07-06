function getApiBaseUrl() {
  if (typeof window === "undefined") {
    return (
      process.env.INTERNAL_API_BASE_URL ??
      process.env.NEXT_PUBLIC_API_BASE_URL ??
      "http://localhost:8000"
    );
  }

  return process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
}

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

export type AgentTurnResponse = {
  agent_id: string;
  agent_name: string;
  agent_response: string;
  inference_mode: string;
};

export type RoundPlayer = {
  id: string;
  name: string;
  kind: "human" | "agent";
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
  imposter_hint: string | null;
  user_role: "player" | "imposter";
  status: string;
  playing_order: RoundPlayer[];
  current_player_id: string | null;
  current_player_name: string | null;
  turns: Turn[];
  created_at: string;
};

export type AgentVote = {
  voter_agent_id: string;
  voter_agent_name: string;
  voted_for: string;
  inference_mode: string;
};

export type VoteCount = {
  player_id: string;
  player_name: string;
  votes: number;
};

export type VoteResult = {
  voted_agent_id: string;
  voted_agent_name: string;
  secret_word: string;
  imposter_was: string;
  agent_votes: AgentVote[];
  vote_counts: VoteCount[];
  group_voted_player_id: string | null;
  group_voted_player_name: string | null;
  imposter_won: boolean;
  round_winner: "players" | "imposter";
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, {
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

export function createRound() {
  return request<Round>("/rounds", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function getRound(roundId: string) {
  return request<Round>(`/rounds/${roundId}`);
}

export function submitRoundClue(roundId: string, clue: string) {
  return request<Round>(`/rounds/${roundId}/clue`, {
    method: "POST",
    body: JSON.stringify({ clue }),
  });
}

export function voteRound(roundId: string, agentId: string) {
  return request<VoteResult>(`/rounds/${roundId}/vote`, {
    method: "POST",
    body: JSON.stringify({ agent_id: agentId }),
  });
}
