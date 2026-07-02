# Project Context

## Current Status

LLMposter has a working local MVP gameplay loop for a word-based imposter game.

The project is intentionally still pre-Docker and pre-database persistence. The immediate goal is to start containerization from a stable service boundary rather than continuing to add gameplay complexity.

## Architecture

Current services:

- `frontend/`: Next.js + TypeScript app.
- `backend/`: FastAPI app.
- local model server: Ollama, using an OpenAI-compatible `/v1/chat/completions` endpoint.

Current runtime path:

```text
browser -> Next.js frontend -> FastAPI backend -> Ollama model server
```

The frontend never calls the model server directly.

## Backend

Important files:

- `backend/app/main.py`
- `backend/app/core/config.py`
- `backend/app/api/routes/agents.py`
- `backend/app/api/routes/rounds.py`
- `backend/app/services/agents.py`
- `backend/app/services/inference.py`
- `backend/tests/test_agents.py`
- `backend/tests/test_rounds.py`

Backend capabilities:

- health endpoint
- agent listing endpoint
- single-agent generation endpoint retained for testing/debugging
- in-memory round store
- round creation
- role assignment
- agent clue generation
- voting
- model inference error handling

Current round model:

- 1 human player
- 4 LLM agents
- imposter is randomly selected from all 5 seats
- if the human is not the imposter, the frontend receives `visible_word`
- if the human is the imposter, the frontend receives `visible_word: null` and `user_role: "imposter"`
- hidden backend state stores the real `secret_word` and `imposter_player_id`

Current endpoints:

```text
GET  /health
GET  /agents
POST /agents/generate
POST /rounds
GET  /rounds/{round_id}
POST /rounds/{round_id}/vote
```

`POST /rounds` currently accepts:

```json
{
  "secret_word": "satellite"
}
```

It immediately queries the four LLM agents and returns opening clues.

`POST /rounds/{round_id}/vote` accepts:

```json
{
  "agent_id": "agent_a"
}
```

The vote response tells the player whether the vote was correct and reveals who the imposter was.

## Model Inference

Local inference uses Ollama.

Expected local backend env:

```env
INFERENCE_MODE=remote
MODEL_SERVER_URL=http://localhost:11434
MODEL_NAME=qwen2.5:1.5b
```

Fake mode is still available:

```env
INFERENCE_MODE=fake
```

Inference design:

- one base model
- four LLM seats
- behavior is controlled through prompt/config, not separate model deployments
- `InferenceClient` sends OpenAI-compatible chat-completions requests
- model failures are converted into `InferenceServiceError`
- API routes convert inference failures into `503`

Current agent prompt behavior:

- if agent receives the word, it is prompted to describe it without using the word
- if agent is the imposter, it receives no word and is prompted for a random natural phrase
- prompt is intentionally short because the local model is small
- output target is strictly 2-5 words

## Frontend

Important files:

- `frontend/src/app/page.tsx`
- `frontend/src/app/game-console.tsx`
- `frontend/src/app/globals.css`
- `frontend/src/lib/api.ts`

Current gameplay UI:

1. Start screen shows only game instructions and a Start Round button.
2. The secret word is currently a hidden frontend default sent to the backend.
3. After starting, the player sees either:
   - their word if they are not the imposter
   - an imposter notification if they are the imposter
4. The player must lock their own clue before seeing LLM clues.
5. After the human clue is locked, the four LLM clues are shown.
6. The player can vote out one LLM agent.
7. The UI shows whether the vote was correct.

Known frontend caveat:

- The frontend still contains `DEFAULT_SECRET_WORD = "satellite"`.
- This is acceptable for the local MVP, but later the backend should own word selection so setup data is never present in the client before the round starts.

## Testing And Validation

Last known validation status:

```text
backend pytest: 15 passed
backend ruff: passed
frontend typecheck: passed
frontend build: passed
```

Commands:

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

## Key Decisions

- Standard agent planning path is `.agent`, not `.agents`.
- Docker was intentionally deferred until after a working local service boundary.
- PostgreSQL persistence is intentionally deferred.
- Current round storage is an in-memory dictionary.
- Current model server is local Ollama.
- Future production model serving should move toward containerized vLLM on cloud GPU.
- Keep the backend as the only component that talks to the model server.
- Stop gameplay expansion here for now and move to containerization/CI/CD learning.

## Next Step

Start the containerization lesson.

Recommended first containerization objective:

```text
Run the FastAPI backend in a Docker container while the frontend and Ollama still run locally.
```

Important networking lesson:

```text
localhost inside a container is not the host machine.
```

When the backend runs in Docker and Ollama runs on the Windows host, the backend will likely need:

```env
MODEL_SERVER_URL=http://host.docker.internal:11434
```

instead of:

```env
MODEL_SERVER_URL=http://localhost:11434
```

Do not begin with full Docker Compose. Start with a single backend Dockerfile and one backend container.
