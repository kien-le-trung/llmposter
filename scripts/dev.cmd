@echo off
setlocal

set "ROOT=%~dp0.."
set "BACKEND_DIR=%ROOT%\backend"
set "FRONTEND_DIR=%ROOT%\frontend"

cd /d "%ROOT%"

echo Starting Docker dependencies: db and model...
docker compose up -d db model
if errorlevel 1 (
  echo Failed to start Docker dependencies.
  exit /b 1
)

if not exist "%BACKEND_DIR%\venv\Scripts\activate.bat" (
  echo Missing backend virtual environment: %BACKEND_DIR%\venv
  echo Create it and install backend dependencies before running this script.
  exit /b 1
)

if not exist "%FRONTEND_DIR%\node_modules" (
  echo Missing frontend dependencies: %FRONTEND_DIR%\node_modules
  echo Run npm.cmd install from the frontend directory before running this script.
  exit /b 1
)

echo Starting backend dev server in a new window...
start "LLMposter Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && call venv\Scripts\activate.bat && set APP_ENV=development&& set BACKEND_CORS_ORIGINS=http://localhost:3000&& set BACKEND_CORS_ORIGIN_REGEX=https?://(localhost^|127\.0\.0\.1)(:\d+)?&& set DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/llmposter&& set LLM_EXPERIMENT_CONFIG=experiments/model_configs/self_hosted_LLM_qwen.json&& set EMBEDDING_MODEL_SERVER_URL=http://localhost:11434&& set EMBEDDING_MODEL_NAME=nomic-embed-text&& set INFERENCE_MODE=remote&& set AGENT_CONFIG_SOURCE=database&& set WORD_SELECTION_MODE=random&& python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo Starting frontend dev server in a new window...
start "LLMposter Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && set NEXT_PUBLIC_API_BASE_URL=http://localhost:8000&& set INTERNAL_API_BASE_URL=http://localhost:8000&& npm.cmd run dev"

echo.
echo Live editing environment started.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:8000
echo Docker dependencies are running: db, model
echo.
echo Stop backend/frontend by closing their terminal windows or pressing Ctrl+C in each.
echo Stop Docker dependencies with: docker compose stop db model
