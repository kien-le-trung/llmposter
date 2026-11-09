"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  Round, VoteResult, createRound, submitDiscussionComment, getRound,
  submitRoundClue, voteRound,
} from "@/lib/api";
import { playerDisplayName } from "@/lib/player-names";

export function GameConsole() {
  const [humanName, setHumanName] = useState("");
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
  const helpRef = useRef<HTMLDialogElement>(null);
  const followMessages = useRef(true);
  const actionVersion = useRef(0);
  const actionPending = useRef(false);

  const roundId = round?.id;
  const roundStatus = round?.status;
  const clueCount = round?.turns.reduce((count, turn) => count + turn.responses.length, 0) ?? 0;
  const messageCount = clueCount + (round?.discussion.messages.length ?? 0);
  const pendingPlayer = round?.status === "generating_clues"
    ? round.playing_order.find((player) => player.id === round.current_player_id && player.kind === "agent")
    : undefined;
  const typingPlayer = pendingPlayer && !round?.turns.some((turn) =>
    turn.responses.some((response) => response.agent_id === pendingPlayer.id),
  ) ? pendingPlayer : undefined;

  useEffect(() => {
    const list = transcriptRef.current;
    if (list && followMessages.current) list.scrollTop = list.scrollHeight;
  }, [roundId, messageCount, typingPlayer?.id]);

  useEffect(() => () => { actionVersion.current += 1; }, []);

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
        if (!cancelled && version === actionVersion.current) {
          setError(caught instanceof Error ? caught.message : "Could not refresh round");
        }
      } finally {
        fetching = false;
      }
    }, 500);
    return () => { cancelled = true; window.clearInterval(interval); };
  }, [roundId, roundStatus]);

  function beginAction() {
    actionPending.current = true;
    setError(null);
    return ++actionVersion.current;
  }

  function finishAction(version: number) {
    if (version !== actionVersion.current) return;
    actionVersion.current += 1;
    actionPending.current = false;
    setIsCreatingRound(false);
    setIsSubmittingClue(false);
    setIsVoting(false);
    setDiscussionBusy(false);
  }

  async function handleCreateRound(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (actionPending.current || !humanName.trim()) return;
    const version = beginAction();
    setIsCreatingRound(true);
    try {
      const created = await createRound(humanName);
      if (version !== actionVersion.current) return;
      followMessages.current = true;
      setRound(created);
      setHumanClueDraft("");
      setCommentDraft("");
      setNow(Date.now());
      setVoteResult(null);
    } catch (caught) {
      if (version === actionVersion.current) {
        setError(caught instanceof Error ? caught.message : "Could not start the round");
      }
    } finally {
      finishAction(version);
    }
  }

  async function handleSubmitHumanClue(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!round || actionPending.current || !humanClueDraft.trim()) return;
    const version = beginAction();
    setIsSubmittingClue(true);
    try {
      const updated = await submitRoundClue(round.id, humanClueDraft.trim());
      if (version !== actionVersion.current) return;
      setRound(updated);
      setHumanClueDraft("");
    } catch (caught) {
      if (version === actionVersion.current) {
        setError(caught instanceof Error ? caught.message : "Could not submit clue");
      }
    } finally {
      finishAction(version);
    }
  }

  const remaining = round?.discussion.deadline
    ? Math.max(0, Math.ceil((Date.parse(round.discussion.deadline) - now) / 1000)) : 0;
  const discussionOpen = round?.status === "discussion" && remaining > 0;

  async function handleDiscussion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!round || !discussionOpen || actionPending.current || !commentDraft.trim()) return;
    const version = beginAction();
    setDiscussionBusy(true);
    try {
      const updated = await submitDiscussionComment(round.id, commentDraft.trim());
      if (version !== actionVersion.current) return;
      setRound(updated);
      setCommentDraft("");
    } catch (caught) {
      if (version === actionVersion.current) {
        setError(caught instanceof Error ? caught.message : "Could not send your comment");
      }
    } finally {
      finishAction(version);
    }
  }

  async function handleVote(agentId: string) {
    if (!round || actionPending.current) return;
    const version = beginAction();
    setIsVoting(true);
    try {
      const result = await voteRound(round.id, agentId);
      if (version !== actionVersion.current) return;
      setVoteResult(result);
      setRound((current) => current ? { ...current, status: "complete" } : current);
    } catch (caught) {
      if (version === actionVersion.current) {
        setError(caught instanceof Error ? caught.message : "Could not submit vote");
      }
    } finally {
      finishAction(version);
    }
  }

  function resetRound() {
    actionVersion.current += 1;
    actionPending.current = false;
    followMessages.current = true;
    setRound(null);
    setHumanClueDraft("");
    setCommentDraft("");
    setVoteResult(null);
    setError(null);
    setIsCreatingRound(false);
    setIsSubmittingClue(false);
    setIsVoting(false);
    setDiscussionBusy(false);
  }

  if (!round) {
    return (
      <section className="welcome-screen" aria-labelledby="welcome-title">
        <p className="eyebrow">The Chameleon, but played with AI.</p>
        <h1 id="welcome-title"><span className="brand-accent">llm</span>poster</h1>
        <p className="welcome-description">One secret word. One imposter. Can you tell who is bluffing?</p>
        <form onSubmit={handleCreateRound} className="welcome-form">
          <label className="field">
            <span>Your name</span>
            <input name="humanName" autoComplete="nickname" required maxLength={40}
              value={humanName} onChange={(event) => setHumanName(event.target.value)}
              placeholder="What should the players call you?" disabled={isCreatingRound} />
          </label>
          <button type="submit" className="start-button" disabled={isCreatingRound || !humanName.trim()}>
            {isCreatingRound ? <><TypingDots /> Starting</> : <>Start <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 12h14M13 6l6 6-6 6" /></svg></>}
          </button>
          <p>Give a clue. Talk it out. Cast your vote.</p>
        </form>
        <button type="button" className="secondary-button" onClick={() => helpRef.current?.showModal()}>How to play</button>
        <dialog ref={helpRef} className="help-dialog" aria-labelledby="help-title">
          <h2 id="help-title">How to play</h2>
          <p>One player is the imposter. Everyone else knows the same secret word.</p>
          <ol>
            <li><strong>Read your briefing.</strong> You receive the secret word, or a hint if you are the imposter. Keep it private.</li>
            <li><strong>Give a clue.</strong> On your turn, share a short clue of 2–5 words. Show you know the word without saying it outright. As the imposter, blend in.</li>
            <li><strong>Talk it out.</strong> During the timed debate, ask questions and share suspicions. You can send comments freely while the AI players discuss.</li>
            <li><strong>Cast your vote.</strong> When discussion ends, choose the player you suspect. Catch the imposter to win; as the imposter, avoid the group vote.</li>
          </ol>
          <form method="dialog"><button type="submit" autoFocus>Got it</button></form>
        </dialog>
        {isCreatingRound && <span className="sr-only" role="status">Starting your round</span>}
        {error && <p className="error" role="alert">{error}</p>}
      </section>
    );
  }

  const displayName = (id: string | null, fallback: string) =>
    playerDisplayName(round.playing_order, id, fallback);
  const identityTitle = (id: string | null, fallback: string) => {
    const original = round.playing_order.find((player) => player.id === id)?.name ?? fallback;
    return original === displayName(id, fallback) ? undefined : `Original name: ${original}`;
  };
  const statusText = isVoting ? "Collecting votes..."
    : round.status === "awaiting_human_clue" ? "Your turn. Share a short clue."
    : round.status === "discussion" ? (discussionOpen ? "Talk it out. Who is bluffing?" : "Discussion is closing...")
    : round.status === "ready_to_vote" ? "Time to vote. Choose the player you suspect."
    : round.status === "complete" ? "Round complete. The votes are in."
    : round.status === "generation_failed" ? "The clues couldn't be completed. Start a new round to try again."
    : "";

  return (
    <>
      <header className="header game-header">
        <h1><span className="brand-accent">llm</span>poster</h1>
        <button type="button" className="secondary-button" onClick={resetRound}>New Round</button>
      </header>
      <section className="game-layout">
        <aside className="table-panel">
          <div className="panel-heading"><p className="eyebrow">At the table</p><h2>Players</h2></div>
          <ul className="player-list">
            {round.playing_order.map((player) => {
              const name = displayName(player.id, player.name);
              const current = round.current_player_id === player.id
                && ["generating_clues", "awaiting_human_clue"].includes(round.status);
              return (
                <li key={player.id} className={current ? "current-player" : undefined}>
                  <span className={`seat-token ${player.kind}`} aria-hidden="true">{name[0]}</span>
                  <div>
                    <strong title={identityTitle(player.id, player.name)}>{name}</strong>
                    <span>{current && player.kind === "human" ? "Your turn" : player.kind === "human" ? "Player" : "AI player"}</span>
                  </div>
                </li>
              );
            })}
          </ul>
        </aside>
        <section className="round-panel" aria-labelledby="round-heading">
          <div className="panel-heading"><p className="eyebrow">Round {round.id.slice(0, 8)}</p><h2 id="round-heading">Find the imposter</h2></div>
          <div className="round-brief">
            <span>Your private briefing</span>
            <strong>{round.user_role === "imposter" ? `You are the imposter. Hint: ${round.imposter_hint}` : `Your word: ${round.visible_word}`}</strong>
          </div>
          <section className="chat-panel" aria-labelledby="conversation-heading">
            <div className="chat-heading conversation-heading">
              <h2 id="conversation-heading">Conversation</h2>
              {round.discussion.deadline && <span className="discussion-timer" role="timer">
                {round.status === "discussion" ? `${remaining}s remaining` : "Discussion closed"}
              </span>}
            </div>
            <ol ref={transcriptRef} className="chat-thread conversation-thread" aria-live="polite"
              aria-relevant="additions" aria-label="Clues and discussion"
              onScroll={(event) => {
                const list = event.currentTarget;
                followMessages.current = list.scrollHeight - list.scrollTop - list.clientHeight < 60;
              }}>
              {round.turns.flatMap((turn) => turn.responses.map((response) => (
                <ChatMessage key={`clue:${turn.id}:${response.agent_id}`} name={displayName(response.agent_id, response.agent_name)}
                  originalName={response.agent_name} human={response.agent_id === "human"}>
                  <p>{response.agent_response}</p>
                </ChatMessage>
              )))}
              {[...round.discussion.messages].sort((a, b) => a.sequence - b.sequence).map((message) => (
                <ChatMessage key={`discussion:${message.id}`} name={displayName(message.player_id, message.player_name)}
                  originalName={message.player_name} human={message.player_id === "human"}>
                  <p>{message.text}</p>
                </ChatMessage>
              ))}
              {messageCount === 0 && !typingPlayer && <li className="chat-empty">Your conversation starts with a clue.</li>}
              {typingPlayer && <ChatMessage key={`typing:${typingPlayer.id}`} name={displayName(typingPlayer.id, typingPlayer.name)} originalName={typingPlayer.name}>
                <div className="typing-indicator" role="status">
                  <TypingDots /><span className="sr-only">Waiting for {displayName(typingPlayer.id, typingPlayer.name)}'s clue</span>
                </div>
              </ChatMessage>}
            </ol>
            <div className="chat-composer">
              {statusText && <p className={`game-status ${round.status === "generation_failed" ? "error" : ""}`} role="status">{statusText}</p>}
              {round.status === "awaiting_human_clue" && <form className="game-form" onSubmit={handleSubmitHumanClue}>
                <label className="field"><span>Your clue</span>
                  <textarea maxLength={200} value={humanClueDraft} disabled={isSubmittingClue}
                    onChange={(event) => setHumanClueDraft(event.target.value)} placeholder="Add your own 2-5 word clue." />
                </label>
                <div className="actions"><button type="submit" disabled={!humanClueDraft.trim() || isSubmittingClue}>
                  {isSubmittingClue ? "Submitting..." : "Lock Clue"}
                </button></div>
              </form>}
              {discussionOpen && <form className="game-form" onSubmit={handleDiscussion}>
                <label className="field"><span>Your comment</span>
                  <textarea maxLength={500} value={commentDraft} disabled={discussionBusy}
                    onChange={(event) => setCommentDraft(event.target.value)} placeholder="Ask a question or share a suspicion..." />
                </label>
                <div className="composer-actions"><span>{commentDraft.length}/500</span>
                  <button type="submit" disabled={discussionBusy || !commentDraft.trim()}>{discussionBusy ? "Sending..." : "Send"}</button>
                </div>
              </form>}
              {!discussionOpen && commentDraft && <div className="unsent-draft"><strong>Unsent draft</strong><p>{commentDraft}</p></div>}
              {round.discussion.error && <p role="alert" className="error">{round.discussion.error}</p>}
            </div>
          </section>
          {(round.status === "ready_to_vote" || round.status === "complete") && <section className="vote-panel">
            <div className="panel-heading"><p className="eyebrow">Vote</p><h2>Choose the imposter</h2></div>
            {voteResult ? <div className={voteResult.imposter_won ? "vote-result imposter" : "vote-result players"}>
              <strong>{voteResult.imposter_won ? "Imposter wins" : "Players win"}</strong>
              <span>Group vote: <span title={identityTitle(voteResult.group_voted_player_id, voteResult.group_voted_player_name ?? "No consensus")}>
                {displayName(voteResult.group_voted_player_id, voteResult.group_voted_player_name ?? "No consensus")}
              </span>.</span>
              <span>Secret word: {voteResult.secret_word}.</span>
              <span>The imposter was <span title={identityTitle(null, voteResult.imposter_was)}>{displayName(null, voteResult.imposter_was)}</span>.</span>
            </div> : <div className="vote-grid">
              {round.playing_order.filter((player) => player.kind === "agent").map((player) => <button
                key={player.id} type="button" className="vote-button" title={identityTitle(player.id, player.name)}
                disabled={isVoting} onClick={() => handleVote(player.id)}>{displayName(player.id, player.name)}</button>)}
            </div>}
            {voteResult && <div className="agent-votes">
              <strong>Vote count</strong><ul>{voteResult.vote_counts.map((count) => <li key={count.player_id}>
                <span title={identityTitle(count.player_id, count.player_name)}>{displayName(count.player_id, count.player_name)}</span>
                <p>{count.votes} vote{count.votes === 1 ? "" : "s"}</p>
              </li>)}</ul>
              <strong>Player votes</strong><ul>{voteResult.agent_votes.map((vote) => <li key={vote.voter_agent_id}>
                <span title={identityTitle(vote.voter_agent_id, vote.voter_agent_name)}>{displayName(vote.voter_agent_id, vote.voter_agent_name)}</span>
                <p title={identityTitle(null, vote.voted_for)}>{displayName(null, vote.voted_for)}</p>
              </li>)}</ul>
            </div>}
          </section>}
          {error && <p className="error" role="alert">{error}</p>}
        </section>
      </section>
    </>
  );
}

function TypingDots() {
  return <span className="typing-dots" aria-hidden="true"><span /><span /><span /></span>;
}

function ChatMessage({ name, originalName, human = false, children }: {
  name: string; originalName: string; human?: boolean; children: React.ReactNode;
}) {
  return <li className={`chat-message ${human ? "human" : "agent"}`}>
    <div className="chat-avatar" aria-hidden="true">{name[0]}</div>
    <div className="chat-bubble">
      <strong title={originalName === name ? undefined : `Original name: ${originalName}`}>{name}</strong>
      {children}
    </div>
  </li>;
}
