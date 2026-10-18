"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Agent,
  Round,
  Turn,
  VoteResult,
  createRound,
  submitDiscussionComment,
  getRound,
  submitRoundClue,
  voteRound,
} from "@/lib/api";

type GameConsoleProps = {
  agents: Agent[];
};

type PlayerSeat = {
  id: string;
  name: string;
  kind: "human" | "agent";
};

export function GameConsole({ agents }: GameConsoleProps) {
  const [round, setRound] = useState<Round | null>(null);
  const [humanClueDraft, setHumanClueDraft] = useState("");
  const [voteResult, setVoteResult] = useState<VoteResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCreatingRound, setIsCreatingRound] = useState(false);
  const [isSubmittingClue, setIsSubmittingClue] = useState(false);
  const [isVoting, setIsVoting] = useState(false);
  const [commentDraft, setCommentDraft] = useState("");
  const [discussionBusy, setDiscussionBusy] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const transcriptRef = useRef<HTMLOListElement>(null);
  const followMessages = useRef(true);
  const actionVersion = useRef(0);
  const actionPending = useRef(false);

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
  const displayedSeats = round
    ? round.playing_order.map((player) => ({
        id: player.id,
        name: player.name,
        kind: player.kind,
      }))
    : seats;
  useEffect(() => {
    const list = transcriptRef.current;
    if (list && followMessages.current) list.scrollTop = list.scrollHeight;
  }, [round?.discussion.messages.length]);

  const roundId = round?.id;
  const roundStatus = round?.status;
  useEffect(() => {
    if (!roundId || !["generating_clues", "discussion"].includes(roundStatus ?? "")) return;
    let cancelled = false;
    let fetching = false;
    const interval = window.setInterval(async () => {
      setNow(Date.now());
      if (fetching || actionPending.current) return;
      fetching = true;
      const version = actionVersion.current;
      try {
        const updated = await getRound(roundId);
        if (!cancelled && version === actionVersion.current) setRound(updated);
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : "Could not refresh round");
      } finally {
        fetching = false;
      }
    }, 500);
    return () => { cancelled = true; window.clearInterval(interval); };
  }, [roundId, roundStatus]);

  async function handleDiscussion() {
    if (!round || actionPending.current) return;
    const version = ++actionVersion.current;
    actionPending.current = true;
    setDiscussionBusy(true);
    setError(null);
    try {
      const updated = await submitDiscussionComment(round.id, commentDraft.trim());
      if (version === actionVersion.current) {
        setRound(updated);
        setCommentDraft("");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Discussion action failed");
    } finally {
      actionVersion.current += 1;
      actionPending.current = false;
      setDiscussionBusy(false);
    }
  }

  const remaining = round?.discussion.deadline
    ? Math.max(0, Math.ceil((Date.parse(round.discussion.deadline) - now) / 1000)) : 0;
  const discussionOpen = round?.status === "discussion" && remaining > 0;

  async function handleCreateRound(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    actionVersion.current += 1;
    setIsCreatingRound(true);
    setError(null);

    try {
      const createdRound = await createRound();
      setRound(createdRound);
      setHumanClueDraft("");
      setCommentDraft("");
      setNow(Date.now());
      setVoteResult(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start the round");
    } finally {
      setIsCreatingRound(false);
    }
  }

  async function handleSubmitHumanClue(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!round) {
      return;
    }

    const trimmedClue = humanClueDraft.trim();
    if (!trimmedClue) {
      return;
    }

    setIsSubmittingClue(true);
    setError(null);
    try {
      const updatedRound = await submitRoundClue(round.id, trimmedClue);
      setRound(updatedRound);
      setHumanClueDraft("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not submit clue");
    } finally {
      setIsSubmittingClue(false);
    }
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
    actionVersion.current += 1;
    setCommentDraft("");
    followMessages.current = true;
    setRound(null);
    setHumanClueDraft("");
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
          {displayedSeats.map((seat) => (
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
                else's clues, discuss freely, then vote out the player you suspect.
              </p>
            </div>
            <button type="submit" disabled={isCreatingRound}>
              {isCreatingRound ? "Starting..." : "Start Round"}
            </button>
          </form>
        ) : (
          <>
            <div className="round-brief">
              <span>Status: {round.status}</span>
              <strong>
                {round.user_role === "imposter"
                  ? `You are the imposter. Hint: ${round.imposter_hint}`
                  : `Your word: ${round.visible_word}`}
              </strong>
            </div>

            <ol className="turn-list" aria-live="polite">
              {round.turns.length > 0 ? (
                round.turns.map((turn) => (
                  <TurnCard
                    key={turn.id}
                    turn={turn}
                  />
                ))
              ) : (
                <li className="empty-turn">
                  {round.current_player_id === "human"
                    ? "You are first. Lock your clue to continue."
                    : "Waiting for opening clues."}
                </li>
              )}
            </ol>

            {round.discussion.enabled && round.discussion.deadline ? (
              <section className="discussion-panel" aria-labelledby="discussion-heading">
                <h2 id="discussion-heading">Discussion</h2>
                <p role="timer">{round.status === "discussion" ? `${remaining}s remaining` : "Discussion closed"}</p>
                <ol ref={transcriptRef} className="discussion-transcript" aria-live="polite" aria-label="Discussion comments"
                  onScroll={(event) => {
                    const list = event.currentTarget;
                    followMessages.current = list.scrollHeight - list.scrollTop - list.clientHeight < 60;
                  }}>
                  {round.discussion.messages.map((message) => (
                    <li key={message.id}><strong>{message.player_name}</strong><p>{message.text}</p></li>
                  ))}
                </ol>
                {round.discussion.messages.length === 0 && <p>No comments yet.</p>}
                {round.discussion.error && <p role="alert" className="error">{round.discussion.error}</p>}
                {discussionOpen && (
                  <form className="game-form" onSubmit={(event) => { event.preventDefault(); void handleDiscussion(); }}>
                    <label className="field"><span>Your comment</span>
                      <textarea maxLength={500} value={commentDraft} onChange={(event) => setCommentDraft(event.target.value)} />
                    </label>
                    <span>{commentDraft.length}/500</span>
                    <div className="actions">
                      <button type="submit" disabled={discussionBusy || !commentDraft.trim()}>Send</button>
                    </div>
                  </form>
                )}
                {!discussionOpen && commentDraft && <div><strong>Unsent draft</strong><p>{commentDraft}</p></div>}
              </section>
            ) : null}

            {round.status === "ready_to_vote" || round.status === "complete" ? (
              <section className="vote-panel">
                <div className="panel-heading">
                  <p className="eyebrow">Vote</p>
                  <h2>Choose the imposter</h2>
                </div>

                {voteResult ? (
                  <div className={voteResult.imposter_won ? "vote-result imposter" : "vote-result players"}>
                    <strong>
                      {voteResult.imposter_won ? "Imposter wins" : "Players win"}
                    </strong>
                    <span>
                      Group vote:{" "}
                      {voteResult.group_voted_player_name ?? "No consensus"}.
                    </span>
                    <span>Secret word: {voteResult.secret_word}.</span>
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

                {voteResult ? (
                  <div className="agent-votes">
                    <strong>Vote count</strong>
                    <ul>
                      {voteResult.vote_counts.map((count) => (
                        <li key={count.player_id}>
                          <span>{count.player_name}</span>
                          <p>{count.votes} vote{count.votes === 1 ? "" : "s"}</p>
                        </li>
                      ))}
                    </ul>
                    <strong>Agent votes</strong>
                    <ul>
                      {voteResult.agent_votes.map((vote) => (
                        <li key={vote.voter_agent_id}>
                          <span>{vote.voter_agent_name}</span>
                          <p>{vote.voted_for}</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </section>
            ) : null}

            {round.status === "awaiting_human_clue" ? (
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
                    disabled={humanClueDraft.trim().length === 0 || isSubmittingClue}
                  >
                    {isSubmittingClue ? "Submitting..." : "Lock Clue"}
                  </button>
                  <button type="button" className="secondary-button" onClick={resetRound}>
                    New Round
                  </button>
                </div>
              </form>
            ) : (
              <div className="actions">
                <button type="button" className="secondary-button" onClick={resetRound}>
                  New Round
                </button>
              </div>
            )}
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
    <li className="chat-panel">
      <div className="chat-heading">
        <span>Opening clues</span>
        <p>{turn.user_prompt}</p>
      </div>
      <div className="chat-thread">
        {turn.responses.map((response, index) => (
          <div
            key={`${response.agent_id}-${index}`}
            className={`chat-message ${response.agent_id === "human" ? "human" : "agent"}`}
          >
            <div className="chat-avatar">
              {response.agent_name.slice(0, 1).toUpperCase()}
            </div>
            <div className="chat-bubble">
              <strong>{response.agent_name}</strong>
              <p>{response.agent_response}</p>
            </div>
          </div>
        ))}
      </div>
    </li>
  );
}
