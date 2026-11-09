# LLMposter

Multi-agent LLM social deduction game with a Next.js frontend, FastAPI backend, PostgreSQL, and configured remote inference providers.

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

Set `DATABASE_URL` in `backend/.env`, `.env.docker`, and `.env` to your Neon
connection string with the `postgresql+psycopg://` driver and `sslmode=require`.
Compose uses `.env.docker`; the host backend uses `backend/.env`.
Do not commit real `.env` files. The Compose stack does not run PostgreSQL locally.

`AGENT_CONFIG_SOURCE=static` uses the built-in agent configs. `AGENT_CONFIG_SOURCE=database` reads active agent configs from PostgreSQL and falls back to static configs if needed.

## Daily Development

Recommended daily workflow:

```text
frontend local
backend local
PostgreSQL on Neon (no local database container)
```

Start the live-editing environment:

```powershell
scripts\dev.cmd
```

This migrates the Neon database configured in `backend/.env`, then opens separate
backend and frontend dev-server windows with autoreload enabled. Local development
writes to that database; use a separate Neon development branch for isolation
from production.

Rounds use a random backend-selected noun by default. For repeatable testing, set:

```env
WORD_SELECTION_MODE=fixed
FIXED_SECRET_WORD=satellite
FIXED_IMPOSTER_HINT=Space, signals, or orbit
```

Run the backend locally (manual equivalent):

```powershell
cd backend
.\venv\Scripts\activate
python -m alembic upgrade head
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

The MVP keeps active round state in backend memory; Neon stores agent
configuration plus clue, vote, and discussion history. Initialization runs Alembic
migrations before seeding. Preserve existing database data; incompatible legacy
schemas require reconciliation rather than deleting tables or volumes.

Chat and embedding inference use the providers configured in `backend/.env`
(or `.env.docker` for Compose). The Compose stack does not include a model service.

## Smoke Checks

Backend health:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Database migration check:

```powershell
docker compose exec backend python -m alembic current
```

## Local Service URLs

From the host machine:

```text
frontend: http://localhost:3000
backend:  http://localhost:8000
postgres: Neon endpoint configured in DATABASE_URL
```

Inside Docker Compose:

```text
frontend -> backend: http://backend:8000
backend -> Neon:     endpoint configured in DATABASE_URL (TLS)
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

Neon data persists independently of Compose containers. Stopping Compose does
not remove the database.

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

Remote chat inference uses the paid OpenRouter model `z-ai/glm-5.3-flash` for
clues, discussion turns, and voting. Configure `OPENROUTER_API_KEY` with available
OpenRouter credits. `LLM_CONFIG` can override the model or provider; explicit local
model configurations and `INFERENCE_MODE=fake` retain their existing behavior.
OpenRouter handles provider routing for the selected model. Transient failures
retain three attempts; structured output also retries schema validation. There is
no application-level secondary chat provider. Embeddings retain their existing
configuration. Restart the backend after changing configuration.

GLM 5.3 Flash was selected for low output cost and broad provider coverage.
Provider prices and availability vary; game-specific latency has not been
benchmarked. See the [OpenRouter model listing](https://openrouter.ai/z-ai/glm-5.3-flash).

## Free-flow discussion

All rounds follow clues -> discussion -> voting. Clues use a private LangGraph pipeline: prepare context -> select strategy -> build prompt -> generate clue -> normalize output. Voting uses structured LLM decisions from each player?s own knowledge and the full public transcript. Discussion lasts 90 seconds
by default (`DISCUSSION_DURATION_SECONDS`, 1-600). Anyone can comment at any time;
there are no raised hands, floor grants, or personalities. Human messages are
saved immediately, independently of AI requests.

Each AI runs a LangGraph debate turn: prepare a private snapshot, generate a
structured memory update plus a comment or `null`, and validate the output.
The backend schedules an opening turn, then reacts to messages from other players
with a 500 ms debounce and a two-second cooldown after each attempt. After ten
seconds of silence (or since its latest completed attempt), an AI can reconsider
whether to speak. An AI's own comment does not trigger an immediate follow-up.

Each AI has at most one turn queued or in flight. All rounds share a four-turn
inference limit. `DISCUSSION_DEBOUNCE_SECONDS`, `DISCUSSION_COOLDOWN_SECONDS`,
`DISCUSSION_SILENCE_SECONDS`, and `DISCUSSION_MAX_CONCURRENT_TURNS` configure these
limits. Replies to older snapshots remain valid while discussion is open;
late, duplicate, or superseded results cannot publish. Failed turns retry after
the cooldown without advancing private memory.

Each player retains bounded private observations, beliefs, questions, and recent
statements for this round. Requests include that memory, the last eight processed
messages, and at most sixteen unseen messages. Catch-up turns update memory while
remaining silent. The backend advances memory only through messages actually
supplied, after successfully persisting any public comment. Voting still uses the
full public transcript. Memory is compressed model output, so it may lose detail;
the persisted public transcript remains authoritative.

Agent memory is process-local, isolated per round and player, and disappears on
restart. There are no persistent graph checkpoints or cross-round learning.
Private graph tracing is disabled. Enable INFO logging for
`app.services.agents.discussion` to collect structured turn logs containing timing,
prompt size, snapshot lag, and result metadata without private prompts or memory.
Latency improvement has not been benchmarked.
The browser polls every 500 ms using ordinary HTTP requests. There are no
WebSockets, token streams, or message queues. Live model latency is unmeasured.

Each AI sees only its own role and word or hint, public identities, clues, and
published discussion. Votes use the configured OpenRouter model.
The timer closes discussion immediately; unsent human text stays as a draft.
New messages scroll into view only when the reader is already near the bottom.

`POST /rounds` retains `include_human: false` for AI-only experiments; these rounds also require discussion and explicitly requested LLM voting. The experiment runner defaults to a 180-second polling timeout. Round responses
include a `discussion` object. `POST /rounds/{id}/discussion` accepts
`{"text":"your comment"}` (up to 500 characters); authorship is determined by the
server. The former action/player-ID parameters and raised-hand/floor fields
have been removed. Existing discussion tables need no migration for free flow.

The backend Docker image runs `python -m alembic upgrade head` before starting
Uvicorn. Migration failures prevent the server from starting. Set the hosting
service's `DATABASE_URL` to the production Neon connection string (using
`postgresql+psycopg://` and retaining its SSL parameters). Alembic and the API use
the same database setting; migrating the local Compose database does not update
Neon. Keep production credentials in the hosting service's environment settings.

If the host overrides the image's default command, use this start command from
the container's `/app` working directory:

```sh
python -m alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

For deployments outside Docker, run migrations from the deployed backend
environment (using its production `DATABASE_URL`) before starting Uvicorn:

```sh
cd backend
python -m alembic upgrade head
python -m alembic current
```

For container deployments the working directory is already `/app`; omit `cd backend`.
Alternatively, use `python -m alembic upgrade head` as the hosting service's
pre-deploy/release command. The backend image includes Alembic and the revision
files. Keep a single backend instance; simultaneous migration attempts are not
coordinated, and active rounds already require a single process.

The first revision adopts the existing agent, clue, and vote tables or creates them
on an empty database. The next revision adds `discussion_messages`. Existing rows
are preserved, and databases previously initialized with `create_all()` need no
manual stamp. Incompatible columns or legacy clue/vote foreign keys stop the
upgrade for manual reconciliation; do not delete production data or use
`alembic stamp head` to bypass the migration. Automatic downgrades are disabled
because the adopted tables may contain pre-existing data.

The existing initializer now runs these migrations before seeding agent configs:

```powershell
cd backend
.\venv\Scripts\python.exe -c "from app.db.seed import init_database; init_database()"
```

Docker's `backend-init` performs the same operation.
Published comments are persisted; active rounds, private agent memories, and deadlines remain
in memory and do not survive a backend restart. Run one backend worker, as with
the existing in-memory round implementation. Unity has not been changed.
