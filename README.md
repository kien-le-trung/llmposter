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

Pull the local model into the model container:

```powershell
docker compose exec model ollama pull qwen2.5:1.5b
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
