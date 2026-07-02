# Step 3: Playable Local Game Loop

## Learning Objective

Turn the current prompt-response demo into a minimal playable round while keeping the system boundaries clean:

- the frontend owns temporary UI state and displays the round transcript
- the backend owns agent configuration and inference orchestration
- the model server remains hidden behind the backend inference client
- persistence is still deferred until the game loop shape is clear

By the end of this step, you should understand how a user action travels through the local distributed application:

```text
browser event -> Next.js client component -> FastAPI route -> inference client -> model server/fake client -> FastAPI response -> browser state update
```

The goal is not a polished game. The goal is one complete, understandable loop that can later be persisted and containerized.

## Current Starting Point

Already done:

- `frontend/src/app/game-console.tsx` can submit a prompt and render one generated response.
- `backend/app/api/routes/agents.py` exposes `/agents` and `/agents/generate`.
- `backend/app/services/inference.py` supports fake and remote inference.
- remote inference failures are converted into controlled backend errors.
- backend tests cover agent listing, generation, bad agent ids, and inference failure.

Current limitation:

- the UI only shows the latest response.
- there is no concept of a round, turn, transcript, or player action history.
- there is no game/session API yet.
- there is no persistence yet.

## Task Definition

Create the first MVP version of a playable round.

For this step, a round means:

1. The user starts with a scenario prompt.
2. The user chooses one agent.
3. The backend generates that agent's response.
4. The frontend appends both the user prompt and agent response to a visible transcript.
5. The user can continue the round by sending another prompt.
6. The user can reset the local round.

This is intentionally frontend-local state for now. Do not add PostgreSQL yet.

## Files To Work In

Likely frontend files:

- `frontend/src/app/game-console.tsx`
- `frontend/src/app/globals.css`
- `frontend/src/lib/api.ts` only if the API types need a small adjustment

Likely backend files:

- no backend changes required unless you discover the response shape is missing essential round data
- avoid creating persistence or game routes during this step

## Implementation Guidance

### 1. Define A Transcript Type

In the client component, define a small local type for transcript entries.

It should capture enough information to render the round history:

- id
- speaker label
- role, such as `user` or `agent`
- text
- optional agent id
- optional inference mode
- timestamp or sequence number

Keep this type local unless it becomes part of the backend API contract.

### 2. Append Instead Of Replace

Right now the UI stores one `result`.

Replace that mental model with a transcript array.

On submit:

- create a user transcript entry
- call `generateAgentResponse`
- create an agent transcript entry from the response
- append both entries in order

Do not clear the transcript after each request.

### 3. Keep Loading And Error States Separate

Keep these as separate pieces of state:

- current prompt text
- selected agent id
- transcript entries
- loading/submitting state
- error message

Do not encode loading or errors as fake transcript entries yet. That can become confusing when persistence is added.

### 4. Add Reset Round

Add a simple reset action that clears:

- transcript
- error

It may keep the current selected agent and prompt draft.

### 5. Avoid Backend Persistence For Now

Do not add database models yet.

The purpose of this step is to learn the game loop shape before deciding what deserves persistence.

Once the transcript behavior feels right, the next step will be to move round/session state behind a backend API and then persist it.

## UX Requirements

The page should make the round state visible:

- show an empty state before the first prompt
- show each user prompt and agent response in order
- show which agent responded
- show whether the response came from `fake` or `remote` inference
- disable submit while a request is in flight
- show a useful error if inference fails
- provide a reset button

Keep the UI compact and functional. Do not spend time on game art or landing-page polish.

## Backend Boundary Requirements

Do not call Ollama or any model server from the frontend.

The only legal path is:

```text
frontend -> FastAPI -> inference client -> model server
```

This protects the future deployment architecture:

- the frontend does not know model server credentials or URLs
- the backend can validate requests
- the backend can log inference metadata
- the backend can later persist game/session state
- the backend can later route between local Ollama and cloud vLLM

## Acceptance Criteria

You are done when:

- a user can submit a prompt and see it appended to a transcript
- the selected agent's response appears below that prompt
- multiple submits build a visible round history
- reset clears the local round history
- inference errors are visible and do not erase the transcript
- backend tests still pass
- frontend typecheck still passes
- frontend build still passes

Validation commands:

```powershell
cd backend
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check app tests
```

```powershell
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

## Learning Questions

Answer these after implementing:

1. What state belongs in the frontend during this step, and why?
2. What state will eventually need to move to the backend?
3. Why should the frontend never call Ollama directly?
4. What data would you need to persist to resume a round later?
5. Which fields in the transcript are UI-only, and which are domain data?

## Stretch Goal

If the basic transcript is done quickly, add a small round summary panel:

- number of user turns
- number of agent responses
- currently selected agent
- current inference mode from the latest response

Do not add persistence as a stretch goal. That is the next formal step.
