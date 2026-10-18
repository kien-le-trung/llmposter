# LLMposter

Multi-agent LLM social deduction game with a Next.js frontend, FastAPI backend, PostgreSQL, and an OpenAI-compatible local model server.

## Project Structure

```text
llmposter/
  frontend/   # Next.js + TypeScript app
  backend/    # FastAPI app
  .agent/     # learning plans and project notes
```

## Environment Files

Local host development:

```powershell
Copy-Item .env.example .env
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.local.example frontend\.env.local
```

Docker Compose development:

```powershell
Copy-Item .env.docker.example .env.docker
```

Do not commit real `.env` files.

`AGENT_CONFIG_SOURCE=static` uses the built-in agent configs. `AGENT_CONFIG_SOURCE=database` reads active agent configs from PostgreSQL and falls back to static configs if needed.

## Daily Development

Recommended daily workflow:

```text
frontend local
backend local
db/model in Docker Compose
```

Start the live-editing environment:

```powershell
scripts\dev.cmd
```

This starts PostgreSQL and Ollama through Docker Compose, then opens separate backend and frontend dev-server windows with autoreload enabled.

Rounds use a random backend-selected noun by default. For repeatable testing, set:

```env
WORD_SELECTION_MODE=fixed
FIXED_SECRET_WORD=satellite
FIXED_IMPOSTER_HINT=Space, signals, or orbit
```

Manual equivalent:

```powershell
docker compose up -d db model
```

Run the backend locally:

```powershell
cd backend
.\venv\Scripts\activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Run the frontend locally:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open:

```text
http://localhost:3000
```

## Full Compose Smoke Test

Run the full local stack:

```powershell
docker compose up -d --build
```

Check containers:

```powershell
docker compose ps
```

Watch logs:

```powershell
docker compose logs -f backend
docker compose logs -f frontend
```

Open:

```text
http://localhost:3000
```

## One-Time Initialization

Initialize database tables and seed agent configs:

```powershell
docker compose exec backend python -c "from app.db.seed import init_database; init_database()"
```

The MVP keeps active round state in backend memory; PostgreSQL stores only agent
configuration plus standalone clue and vote history. After changing from the older
database-backed round schema, recreate the development database with
`docker compose down -v` before initializing it again. `create_all()` does not remove
the obsolete `rounds` table or its foreign keys from an existing database.

Pull the local chat and embedding models into the model container:

```powershell
docker compose exec model ollama pull qwen2.5:1.5b
docker compose exec model ollama pull nomic-embed-text
```

## Smoke Checks

Backend health:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Database seed check:

```powershell
docker compose exec db psql -U postgres -d llmposter -c "select count(*) from agent_configs;"
```

Model check:

```powershell
docker compose exec model ollama list
```

## Local Service URLs

From the host machine:

```text
frontend: http://localhost:3000
backend:  http://localhost:8000
postgres: localhost:5432
ollama:   http://localhost:11434
```

Inside Docker Compose:

```text
frontend -> backend: http://backend:8000
backend -> db:       db:5432
backend -> model:    http://model:11434
```

Browser-side frontend code must use a host-reachable backend URL such as:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Server-side Next.js code inside the frontend container can use:

```env
INTERNAL_API_BASE_URL=http://backend:8000
```

## Useful Commands

Stop containers but keep volumes:

```powershell
docker compose down
```

Stop containers and delete database/model volumes:

```powershell
docker compose down -v
```

Run backend tests locally:

```powershell
cd backend
.\venv\Scripts\python.exe -m pytest
.\venv\Scripts\python.exe -m ruff check app tests
```

Run frontend checks locally:

```powershell
cd frontend
npm.cmd run typecheck
npm.cmd run build
```

## Chat inference providers

For remote chat inference, the backend tries Grok (`grok-4.7`) using
`GROK_API_KEY` from `backend/.env` first. If Grok fails (including a missing
key, HTTP errors, timeouts, or unusable output), it falls back to the existing
OpenRouter `LLM_CONFIG`, using `OPENROUTER_API_KEY`. Transient failures retain
three attempts per provider; structured output also retries schema validation
before falling back. Both keys must be configured for both services to work.

`GROK_CONFIG` optionally overrides the primary provider configuration as JSON;
set it to `null` to use `LLM_CONFIG` alone. Explicit local model configurations
and `INFERENCE_MODE=fake` retain their existing behavior. Embeddings continue
using the existing Gemini configuration. Restart the backend after changing
configuration.

## Free-flow discussion

Human rounds follow clues -> discussion -> voting. Discussion lasts 90 seconds
by default (`DISCUSSION_DURATION_SECONDS`, 1-600). Anyone can comment at any time;
there are no raised hands, floor grants, or personalities. Human messages are
saved immediately, independently of AI requests.

Each AI has a simple loop: read its permitted conversation context, generate a
comment or `null` for silence, publish any comment, wait two seconds, and repeat.
Each AI has at most one request in flight. Responses based on slightly older
conversation context are accepted; responses arriving after the deadline are
discarded. Failed iterations are logged and retried after the same pause.
The browser polls every 500 ms using ordinary HTTP requests. There are no
WebSockets, token streams, or message queues. Live Grok latency is unmeasured.

Each AI sees only its own role and word or hint, public identities, clues, and
published discussion. Discussion-aware votes retain the Grok/OpenRouter fallback.
The timer closes discussion immediately; unsent human text stays as a draft.
New messages scroll into view only when the reader is already near the bottom.

`POST /rounds` accepts `discussion_enabled: false` for the original clue/ML-vote
flow. AI-only rounds keep that flow and cannot enable discussion.
`DISCUSSION_ENABLED=false` changes the default for human rounds. Round responses
include a `discussion` object. `POST /rounds/{id}/discussion` accepts
`{"text":"your comment"}` (up to 500 characters); authorship is determined by the
server. The former action/player-ID parameters and raised-hand/floor fields
have been removed. Existing discussion tables need no migration for free flow.

Before using this feature with an existing database, run the existing initializer
to create the additive `discussion_messages` table:

```powershell
cd backend
.\venv\Scripts\python.exe -c "from app.db.seed import init_database; init_database()"
```

Docker's `backend-init` performs the same operation.
Published comments are persisted; active rounds, raises, and deadlines remain
in memory and do not survive a backend restart. Run one backend worker, as with
the existing in-memory round implementation. Unity has not been changed.
