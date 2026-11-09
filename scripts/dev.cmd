@echo off
setlocal

set "ROOT=%~dp0.."
set "BACKEND_DIR=%ROOT%\backend"
set "FRONTEND_DIR=%ROOT%\frontend"

cd /d "%ROOT%"

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

echo Migrating the database configured in backend/.env...
"%BACKEND_DIR%\venv\Scripts\python.exe" -m alembic -c "%BACKEND_DIR%\alembic.ini" upgrade head
if errorlevel 1 (
  echo Database migration failed. Check backend/.env DATABASE_URL and Neon connectivity.
  exit /b 1
)

echo Starting backend dev server in a new window...
start "LLMposter Backend" cmd /k "cd /d ""%BACKEND_DIR%"" && call venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo Starting frontend dev server in a new window...
start "LLMposter Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && set NEXT_PUBLIC_API_BASE_URL=http://localhost:8000&& set INTERNAL_API_BASE_URL=http://localhost:8000&& npm.cmd run dev"

echo.
echo Live editing environment started.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:8000
echo Database: Neon, configured through backend/.env
echo.
echo Stop backend/frontend by closing their terminal windows or pressing Ctrl+C in each.
