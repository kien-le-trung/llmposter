"use client";

import { FormEvent, useMemo, useState } from "react";
import { Agent, Round, Turn, VoteResult, createRound, voteRound } from "@/lib/api";

type GameConsoleProps = {
  agents: Agent[];
};

type PlayerSeat = {
  id: string;
  name: string;
  kind: "human" | "agent";
};

const DEFAULT_SECRET_WORD = "satellite";

export function GameConsole({ agents }: GameConsoleProps) {
  const [round, setRound] = useState<Round | null>(null);
  const [secretWord, setSecretWord] = useState(DEFAULT_SECRET_WORD);
  const [humanClueDraft, setHumanClueDraft] = useState("");
  const [humanClue, setHumanClue] = useState<string | null>(null);
  const [voteResult, setVoteResult] = useState<VoteResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCreatingRound, setIsCreatingRound] = useState(false);
  const [isVoting, setIsVoting] = useState(false);

  const seats = useMemo<PlayerSeat[]>(
    () => [
      { id: "you", name: "You", kind: "human" },
      ...agents.map((agent) => ({
        id: agent.id,
        name: agent.name,
        kind: "agent" as const,
      })),
    ],
    [agents],
  );

  async function handleCreateRound(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedSecretWord = secretWord.trim();
    if (!trimmedSecretWord) {
      return;
    }

    setIsCreatingRound(true);
    setError(null);

    try {
      const createdRound = await createRound(trimmedSecretWord);
      setRound(createdRound);
      setHumanClueDraft("");
      setHumanClue(null);
      setVoteResult(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start the round");
    } finally {
      setIsCreatingRound(false);
    }
  }

  function handleSubmitHumanClue(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const trimmedClue = humanClueDraft.trim();
    if (!trimmedClue) {
      return;
    }

    setError(null);
    setHumanClue(trimmedClue);
    setHumanClueDraft("");
  }

  async function handleVote(agentId: string) {
    if (!round) {
      return;
    }

    setIsVoting(true);
    setError(null);

    try {
      const result = await voteRound(round.id, agentId);
      setVoteResult(result);
      setRound((currentRound) =>
        currentRound ? { ...currentRound, status: "complete" } : currentRound,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not submit vote");
    } finally {
      setIsVoting(false);
    }
  }

  function resetRound() {
    setRound(null);
    setHumanClueDraft("");
    setHumanClue(null);
    setVoteResult(null);
    setError(null);
  }

  return (
    <section className="game-layout">
      <aside className="table-panel">
        <div className="panel-heading">
          <p className="eyebrow">Table</p>
          <h2>Players</h2>
        </div>

        <ul className="player-list">
          {seats.map((seat) => (
            <li key={seat.id}>
              <span className={`seat-token ${seat.kind}`}>
                {seat.name.slice(0, 1).toUpperCase()}
              </span>
              <div>
                <strong>{seat.name}</strong>
                <span>{seat.kind === "human" ? "Player" : "Candidate"}</span>
              </div>
            </li>
          ))}
        </ul>
      </aside>

      <section className="round-panel">
        <div className="panel-heading">
          <p className="eyebrow">{round ? `Round ${round.id.slice(0, 8)}` : "New round"}</p>
          <h2>{round ? "Find the imposter" : "Set the table"}</h2>
        </div>

        {!round ? (
          <form className="game-form start-screen" onSubmit={handleCreateRound}>
            <div className="instructions">
              <strong>How to play</strong>
              <p>
                Five players sit at the table: you and four candidates. Most players
                receive the same secret word. One player is the imposter and must
                blend in without knowing it. Give a short clue, compare everyone
                else's clues, then vote out the player you suspect.
              </p>
            </div>
            <button type="submit" disabled={isCreatingRound || secretWord.trim().length === 0}>
              {isCreatingRound ? "Starting..." : "Start Round"}
            </button>
          </form>
        ) : (
          <>
            <div className="round-brief">
              <span>Status: {round.status}</span>
              <strong>
                {round.user_role === "imposter"
                  ? "You are the imposter. Blend in without the word."
                  : `Your word: ${round.visible_word}`}
              </strong>
            </div>

            <ol className="turn-list" aria-live="polite">
              {humanClue ? (
                <li className="turn-card human-clue">
                  <div className="turn-answer">
                    <div className="candidate-response">
                      <strong>You</strong>
                      <p>{humanClue}</p>
                    </div>
                  </div>
                </li>
              ) : null}
              {humanClue ? (
                round.turns.map((turn) => <TurnCard key={turn.id} turn={turn} />)
              ) : (
                <li className="empty-turn">
                  Lock your clue before the candidates reveal theirs.
                </li>
              )}
            </ol>

            {humanClue ? (
              <section className="vote-panel">
                <div className="panel-heading">
                  <p className="eyebrow">Vote</p>
                  <h2>Choose the imposter</h2>
                </div>

                {voteResult ? (
                  <div className={voteResult.correct ? "vote-result correct" : "vote-result wrong"}>
                    <strong>{voteResult.correct ? "Correct vote" : "Wrong vote"}</strong>
                    <span>The imposter was {voteResult.imposter_was}.</span>
                  </div>
                ) : (
                  <div className="vote-grid">
                    {agents.map((agent) => (
                      <button
                        key={agent.id}
                        type="button"
                        className="vote-button"
                        disabled={isVoting}
                        onClick={() => handleVote(agent.id)}
                      >
                        {agent.name}
                      </button>
                    ))}
                  </div>
                )}
              </section>
            ) : null}

            <form className="game-form" onSubmit={handleSubmitHumanClue}>
              <label className="field">
                <span>Your clue</span>
                <textarea
                  value={humanClueDraft}
                  onChange={(event) => setHumanClueDraft(event.target.value)}
                  placeholder="Add your own 2-5 word clue."
                />
              </label>

              <div className="actions">
                <button
                  type="submit"
                  disabled={humanClueDraft.trim().length === 0}
                >
                  Lock Clue
                </button>
                <button type="button" className="secondary-button" onClick={resetRound}>
                  New Round
                </button>
              </div>
            </form>
          </>
        )}

        {error ? (
          <p className="error" role="alert">
            {error}
          </p>
        ) : null}
      </section>
    </section>
  );
}

function TurnCard({ turn }: { turn: Turn }) {
  return (
    <li className="turn-card">
      <div className="turn-prompt">
        <span>Opening clues</span>
        <p>{turn.user_prompt}</p>
      </div>
      <div className="turn-answer">
        {turn.responses.map((response) => (
          <div key={response.agent_id} className="candidate-response">
            <strong>{response.agent_name}</strong>
            <p>{response.agent_response}</p>
          </div>
        ))}
      </div>
    </li>
  );
}
