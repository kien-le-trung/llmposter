# LLMposter

Multi-agent LLM game MVP scaffold with separate Next.js frontend and FastAPI backend.

This skeleton intentionally does not include Docker or Docker Compose yet. Runtime boundaries are still explicit so containerization can be added later without restructuring the app.

## Project Structure

```text
llmposter/
  frontend/   # Next.js + TypeScript app
  backend/    # FastAPI app
  .agent/     # project plans and agent notes
```

## Local Setup

The root `.env.example` documents the full local configuration. Each service also has its own example file for the values it reads directly.

Copy backend environment values before running the API:

```powershell
Copy-Item backend\.env.example backend\.env
```

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn app.main:app --reload --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

### Frontend

PowerShell may block `npm.ps1` on this machine, so use `npm.cmd`:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open `http://localhost:3000`.

The frontend defaults to `http://localhost:8000` for API calls. If you need to override it:

```powershell
Copy-Item frontend\.env.local.example frontend\.env.local
```

## Current Runtime Model

The backend exposes a fake inference client by default:

```env
INFERENCE_MODE=fake
```

This keeps the app runnable before self-hosted inference is implemented. Later, replace the fake client behind `backend/app/services/inference.py` with a vLLM or llama.cpp HTTP client using `MODEL_SERVER_URL`.

## No Docker Yet

Do not add Dockerfiles or Compose files during the initial scaffold phase. The near-term goal is to stabilize:

- frontend/backend API contract
- backend inference client boundary
- agent configuration shape
- local run commands

Containerization can wrap these services later.
