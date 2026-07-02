# LLMposter Learning Plan

## Purpose

Use the Phase 0 MVP to learn ML engineering, model hosting, containerization, distributed systems, and networking while still shipping a resume-visible product.

The project now has its first service backbone in place:

- `frontend/` contains a Next.js + TypeScript app.
- `backend/` contains a FastAPI app.
- backend dependencies are installed in `backend/venv`.
- frontend dependencies are installed in `frontend/node_modules`.
- the backend exposes health and agent endpoints.
- the backend has a fake inference boundary in `backend/app/services/inference.py`.
- The backend has completed first-pass Docker containerization.
- Docker Compose is the current learning focus.

This plan assumes prior experience with:

- React Native
- React-style web application development
- FastAPI basics
- PostgreSQL and Supabase-hosted databases

Because of that background, learning time should focus less on CRUD, UI construction, and basic REST patterns, and more on the production mechanics around self-hosted inference and multi-service deployment.

## Phase 0 Learning Objective

By the end of Phase 0, the project should not only be playable. It should also demonstrate that the application is composed of independently understandable services with clear runtime boundaries:

- Next.js frontend
- FastAPI backend
- PostgreSQL database
- self-hosted lightweight LLM inference service
- local development commands for each service
- later Docker Compose environment wiring those services together
- CI/CD path for building, testing, and deploying

The most important learning outcome is understanding how a model server becomes part of a production application, rather than treating the LLM as a black-box API. Docker remains important, but it should wrap stable service boundaries instead of becoming the first source of complexity.

## Current Backbone Status

Completed:

- Separate frontend and backend folders exist.
- Next.js production build succeeds.
- TypeScript typecheck succeeds.
- FastAPI tests pass.
- Agent behavior is represented as configuration.
- The backend has a fake inference client that can be replaced by a real model client.
- The backend has a real OpenAI-compatible model client tested locally through Ollama.
- The backend has five prompt/configured candidate agents sharing one model server.
- The backend has an in-memory round API for shaping game/session state before persistence.
- The MVP gameplay loop supports start round, role reveal, LLM clue generation, human clue entry, and agent voting.
- The FastAPI backend can run as a Docker container.
- The FastAPI backend can run through Docker Compose.
- SQLAlchemy database scaffolding exists for `agent_configs`.
- Environment examples exist for root, frontend, and backend.

Not completed yet:

- PostgreSQL service in Docker Compose.
- PostgreSQL-backed game/session persistence.
- structured inference logging.
- model timeout and error handling.
- CI checks.
- deployment.
- database-backed agent config reads.

Immediate learning focus:

- add PostgreSQL as a second Docker Compose service.
- keep API and inference boundaries explicit.
- keep full gameplay persistence deferred until the database wiring is understood.
- document decisions as they are made.

## Learning Priorities

### 1. Model Hosting And Inference Infrastructure

Focus questions:

- How does the backend call a local or remote model server?
- What request and response contract should the inference service expose?
- How do latency, max tokens, sampling parameters, and concurrency affect the game loop?
- What information should be logged for each inference request?
- How do five prompted agents share one loaded base model safely?

Project tasks:

- Keep the fake inference client as the local development baseline.
- Run one lightweight open-source model through vLLM or llama.cpp after the game loop is useful.
- Put the model server behind a stable HTTP interface.
- Keep one backend inference client module instead of scattering model calls through route handlers.
- Store each agent as configuration: system prompt, temperature, top_p, max_tokens, and version.
- Add timeout handling and clear error behavior for model calls.
- Log inference duration, model name, agent id, token settings, and failure reason.

Concepts to learn:

- model server lifecycle
- model loading cost
- inference request schema
- batching and concurrency basics
- GPU vs CPU inference tradeoffs
- context length and token limits
- OpenAI-compatible inference APIs

Suggested acceptance checks:

- The app can run end-to-end with the fake inference client.
- The app can switch between a fake inference client and the real model server.
- The backend can survive model server downtime with a controlled error.
- All five agents use one base model with different prompt/runtime settings.
- A single inference request can be traced from frontend action to backend route to model server response.

### 2. Local Service Development Before Docker

Focus questions:

- What is the contract between frontend and backend?
- What is the contract between backend routes and the inference service?
- Which runtime values need to come from environment variables?
- Which pieces of state should be persisted before model serving gets complicated?
- Which checks should run locally before deployment or containerization?

Project tasks:

- Make the frontend submit a prompt and render the generated backend response.
- Add a minimal game/session model.
- Add database access behind a small backend persistence boundary.
- Keep backend route handlers thin.
- Add tests for the agent generation endpoint and future game/session endpoints.
- Add a small `GET /health` check that can later grow to include database readiness.
- Keep local run commands accurate for `frontend/` and `backend/`.

Concepts to learn:

- service-oriented local development
- API contracts
- frontend runtime configuration
- backend settings management
- thin route handlers
- testable service boundaries
- local validation before infrastructure work

Suggested acceptance checks:

- The frontend can show backend health and generated agent output.
- Backend tests cover happy path and invalid agent ids.
- The backend can be started from `backend/venv`.
- The frontend can be started with `npm.cmd run dev`.
- No app code depends on Docker-specific hostnames yet.

### 3. Docker Containerization

Focus questions:

- What belongs inside each container image?
- What should be configured at runtime through environment variables?
- How do containers discover each other on a Docker network?
- What data should persist across container restarts?
- How do development containers differ from production containers?

Project tasks:

- Start Docker only after the local frontend/backend/game loop is stable.
- Start with PostgreSQL and model server in Docker.
- Containerize FastAPI after the local backend API and persistence boundaries are stable.
- Containerize Next.js only after the frontend/backend contract is settled.
- Build a final `docker-compose.yml` with frontend, backend, database, and model services.
- Add health checks for backend, database, and model server where practical.
- Use named volumes for PostgreSQL data and model cache when appropriate.

Concepts to learn:

- Dockerfile layering
- build context
- image vs container
- bind mount vs named volume
- Compose service names as DNS names
- container startup order vs service readiness
- environment variable injection
- CPU/GPU container constraints

Suggested acceptance checks:

- `docker compose up` can start the complete local stack.
- The backend connects to PostgreSQL using the Compose service name, not `localhost`.
- The backend connects to the model server using the Compose service name, not `localhost`.
- Restarting the backend container does not erase database state.
- Environment-specific values are supplied through `.env` files or deployment secrets, not hardcoded.

### 4. Distributed Systems And Service Boundaries

Focus questions:

- What failures happen when services communicate over the network?
- Which operations should be synchronous, and which might eventually become queued?
- Where should retries happen, and where should they be avoided?
- How do timeouts protect the user experience?
- What state belongs in the database instead of memory?

Project tasks:

- Treat frontend, backend, database, and model server as separate networked services even during local development.
- Define explicit request boundaries between frontend/backend and backend/model server.
- Add backend timeouts for database and inference calls.
- Return useful API errors when the model server is unavailable or slow.
- Persist game/session state before or around inference so failed generations do not corrupt game flow.

Concepts to learn:

- service boundary design
- network failure modes
- timeouts and retries
- idempotency basics
- stateful vs stateless services
- readiness and liveness
- graceful degradation
- eventual migration to async workers

Suggested acceptance checks:

- Killing the model server produces a predictable backend response.
- Slow inference does not hang indefinitely.
- Game state remains consistent after an inference failure.
- The API contract makes it clear which service owns which responsibility.

### 5. Networking For Multi-Service Apps

Focus questions:

- Why does `localhost` mean different things inside and outside a container?
- How do Docker Compose service names resolve?
- Which ports need to be exposed to the host, and which should remain internal?
- How should CORS be configured between frontend and backend?
- How do deployment URLs replace local Compose DNS?

Project tasks:

- Document local host ports and internal container ports.
- Keep database and model ports internal unless host access is needed for debugging.
- Configure frontend API base URL separately for browser runtime and container runtime if needed.
- Configure CORS explicitly for local and deployed frontend origins.
- Write a short networking note after Docker Compose works.

Concepts to learn:

- host network vs bridge network
- container DNS
- port publishing
- internal service ports
- CORS
- reverse proxy basics
- deployment environment variables

Suggested acceptance checks:

- You can explain why the browser may call `localhost:8000` while the backend container calls `model:8000`.
- Only necessary ports are published from Compose to the host.
- The deployed frontend uses the deployed backend URL, not a local address.
- CORS failures can be diagnosed from browser dev tools and backend config.

### 6. CI/CD And Deployment Infrastructure

Focus questions:

- Which checks should run before deployment?
- How are Docker images built and deployed?
- Where do secrets live?
- How can the deployed app prove that the backend and model path are healthy?

Project tasks:

- Add basic GitHub Actions checks for backend and frontend.
- Add a Docker build check for containerized services.
- Deploy the MVP to Render, Railway, Fly.io, or a similar platform.
- Add a backend health endpoint.
- Add a model health or readiness check if the model server supports it.
- Document deployment environment variables.

Concepts to learn:

- CI vs CD
- build-time vs runtime config
- deployment secrets
- health checks
- release rollback basics
- managed platform networking

Suggested acceptance checks:

- A pull request or push runs automated checks.
- Deployment does not require local manual commands beyond normal platform setup.
- The deployed backend health check can confirm database connectivity.
- The deployed app has a documented model-serving path, even if the model host changes later.

## Suggested Learning Sequence

### Step 1: Local Service Skeleton

Build the minimum frontend and backend flow, but avoid spending too much time on UI polish. Confirm the backend API shape and game loop.

Status: complete.

Primary learning focus:

- service boundaries
- API contracts
- where inference fits into the backend

### Step 2: Fake Inference Client

Create a fake inference implementation before the real model server. This lets the app progress while making the model boundary explicit.

Status: complete. The backend has a fake inference client, and the frontend can render generated fake inference results.

Primary learning focus:

- dependency boundaries
- request/response contracts
- controlled failure behavior

### Step 3: Playable Local Game Loop

Use the existing frontend/backend skeleton to make one complete local round work with fake inference.

Status: complete. The frontend now starts backend rounds, displays role/word reveal, renders LLM clues, records the human clue, and supports voting.

Instruction file: `.agent/learning/STEP_3_PLAYABLE_LOCAL_GAME_LOOP.md`

Primary learning focus:

- frontend/backend request flow
- game state shape
- response rendering
- API error handling
- tests around user-visible behavior

### Step 4: Backend Docker Container

Containerize the FastAPI backend while the frontend and Ollama still run locally.

Status: complete.

Instruction file: `.agent/learning/STEP_4_BACKEND_DOCKER_CONTAINER.md`

Primary learning focus:

- Dockerfile design
- image vs container
- port publishing
- runtime environment variables
- host-to-container networking
- `host.docker.internal` for host Ollama

### Step 5: Docker Compose Backend And Env Files

Run the backend with Docker Compose while the frontend and Ollama remain local.

Status: complete.

Instruction file: `.agent/learning/STEP_5_DOCKER_COMPOSE_BACKEND_ENV.md`

Primary learning focus:

- Docker Compose service definitions
- env-file workflow
- repeatable backend container runs
- Compose logs and lifecycle commands
- host-to-container model networking
- `docker compose up`, logs, and shutdown

### Step 6: Docker Compose With PostgreSQL

Run backend and PostgreSQL together with Docker Compose.

Status: complete.

Instruction file: `.agent/learning/STEP_6_DOCKER_COMPOSE_POSTGRES.md`

Primary learning focus:

- multi-service Compose orchestration
- Compose DNS service names
- named volumes
- PostgreSQL health checks
- database initialization from the backend container

### Step 7: Database-Backed Agent Configs

Read agent configs from PostgreSQL instead of the in-memory config dictionary.

Status: complete.

Primary learning focus:

- repository/service boundary design
- database-backed configuration
- keeping fallback seed data simple

### Step 8: Containerized LLM Inference

Run an OpenAI-compatible model server as a Docker Compose service and point the backend inference client at it.

Status: complete.

Instruction file: `.agent/learning/STEP_8_CONTAINERIZED_LLM_INFERENCE.md`

Primary learning focus:

- model hosting
- inference latency
- model server APIs
- sampling parameters
- model cache volumes
- backend-to-model Compose networking

### Step 9: Frontend Containerization

Containerize Next.js and run the frontend through Docker Compose.

Status: complete.

Instruction file: `.agent/learning/STEP_9_FRONTEND_CONTAINERIZATION.md`

Primary learning focus:

- frontend build images
- browser-facing vs container-internal URLs
- `NEXT_PUBLIC_*` runtime implications
- full local Compose stack

### Step 10: Full Compose Cleanup

Clean up service names, env examples, health checks, and local run documentation.

Status: complete.

Instruction file: `.agent/learning/STEP_10_FULL_COMPOSE_CLEANUP.md`

Primary learning focus:

- repeatable local setup
- service readiness
- developer documentation
- Compose profiles later

### Step 11: CI Checks

Add automated checks before deployment.

Status: next.

Primary learning focus:

- backend tests and lint
- frontend typecheck and build
- Docker image build checks

### Step 12: Deployment

Deploy the MVP.

Status: pending.

Primary learning focus:

- production configuration
- secrets
- health checks
- deployment workflow

## What To Spend Less Time On

Do not over-optimize these areas during Phase 0:

- advanced frontend architecture
- elaborate animations or UI polish
- complex database schema design beyond game/session persistence
- Supabase-specific features unless they directly unblock deployment
- fine-tuning or model evaluation pipelines
- Kubernetes
- multi-model routing
- asynchronous worker queues

These become more useful after the MVP proves the product and service boundaries.

## Notes To Capture While Building

Create short notes as decisions are made:

- why vLLM or llama.cpp was chosen
- exact base model and quantization choice
- local ports and later Compose service names
- model request schema
- timeout values
- known local vs deployed networking differences
- deployment platform constraints
- commands required to run the app locally
- why Docker was deferred until after the local service backbone

These notes can later become `PROJECT_CONTEXT.md`, `DECISIONS.md`, or README deployment documentation.

## Resume-Oriented Outcomes

By the end of this learning plan, the project should support a narrative like:

> Built and deployed a containerized multi-service LLM application with a self-hosted lightweight model server, explicit backend inference client, Docker Compose orchestration, service health checks, and production deployment configuration.

Stronger technical talking points:

- separate frontend/backend service backbone with local validation
- one base model serving five prompted agents
- backend-to-model HTTP inference boundary
- Docker Compose networking across frontend, backend, database, and model services
- controlled timeout and error handling for inference failures
- deployment configuration using environment variables and health checks

## Completion Checklist

- [x] Separate `frontend/` and `backend/` service folders exist.
- [x] Frontend dependencies are installed.
- [x] Backend dependencies are installed in `backend/venv`.
- [x] Backend tests pass.
- [x] Frontend typecheck passes.
- [x] Frontend production build passes.
- [x] Backend has a fake inference client.
- [x] Backend has a real model inference client.
- [x] Agent behavior is configuration-driven rather than separate model-driven.
- [x] Frontend renders generated agent responses.
- [x] Minimal playable local game loop exists.
- [x] Backend has an in-memory round/session API.
- [ ] Game/session state is persisted.
- [ ] PostgreSQL runs in Docker during local development.
- [ ] Model server runs in Docker during local development.
- [x] FastAPI runs successfully as a container.
- [ ] Next.js runs successfully as a container.
- [ ] Full stack runs with Docker Compose.
- [ ] Backend uses Compose service names for database and model connections.
- [x] Inference calls have timeouts and useful error responses.
- [ ] Health checks exist for the backend and database path.
- [ ] Deployment environment variables are documented.
- [ ] CI runs basic checks and at least one Docker build check.
- [ ] A short networking note explains local host ports, container ports, and service DNS.
