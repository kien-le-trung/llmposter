
# Phase 0 — Resume-Visible MVP

## Goal

Build and deploy a complete, working multi-agent LLM application as quickly as possible while learning the fundamentals of deployment and model serving. The emphasis is on shipping a usable product rather than maximizing model performance.

---

# System Architecture

```
                ┌──────────────────────┐
                │      Next.js App     │
                │  (Frontend / UI)     │
                └──────────┬───────────┘
                           │
                    HTTP / REST API
                           │
                ┌──────────▼───────────┐
                │      FastAPI         │
                │  Backend / Game API  │
                └───────┬───────┬──────┘
                        │       │
             PostgreSQL │       │ Model Requests
                        │       │
          ┌─────────────▼──┐  ┌─▼────────────────────┐
          │ PostgreSQL DB  │  │ vLLM / llama.cpp     │
          │ Users/Games    │  │ Qwen2.5-1.5B         │
          └────────────────┘  └─────────────────────┘
```

---

# Recommended Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Next.js + TypeScript |
| Backend | FastAPI |
| Database | PostgreSQL |
| ORM | SQLAlchemy / SQLModel |
| Model Server | vLLM (preferred) or llama.cpp |
| Base Model | Qwen2.5-1.5B-Instruct |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

# Why Self-Hosted Models?

Instead of relying on commercial APIs, use a lightweight open-source model.

Advantages:

- Much lower inference cost
- Learn real inference infrastructure
- Complete control over deployment
- Easy transition to LoRA fine-tuning later
- More realistic ML Engineering experience

The game only requires short, controlled responses, making a 0.5B–1.5B parameter model sufficient.

---

# Agent Design

Do **not** deploy five separate models.

Instead:

```
One base model
        │
        ├── Agent A (Prompt Variant)
        ├── Agent B (Prompt Variant)
        ├── Agent C (Prompt Variant)
        ├── Agent D (Prompt Variant)
        └── Agent E (Prompt Variant)
```

Each agent stores different configuration values such as:

- system prompt
- temperature
- top_p
- max_tokens
- version

This provides five distinct personalities while only loading one model into memory.

---

# Local Development

During early development, use:

- Next.js (`npm run dev`)
- FastAPI (`uvicorn`)
- PostgreSQL (Docker container)
- vLLM or llama.cpp (Docker container)

As the project stabilizes, containerize the backend and frontend as well.

---

# Why Containerize?

In previous projects, the frontend and backend may simply have existed as two folders in one repository.

Containerization is the next step toward production engineering.

Each service has different dependencies:

| Service | Runtime |
|----------|---------|
| Frontend | Node.js |
| Backend | Python |
| Database | PostgreSQL |
| Model Server | CUDA + PyTorch + vLLM |

Running each service inside its own container provides:

- Reproducible environments
- Isolation of dependencies
- Easier deployment
- Independent restarts
- Clear service boundaries
- A production-like architecture

You do **not** need to containerize everything immediately.

A recommended progression is:

## Stage A

- Next.js runs locally
- FastAPI runs locally
- PostgreSQL in Docker
- Model server in Docker

## Stage B

- Containerize FastAPI

## Stage C

- Containerize Next.js

## Stage D

Run the complete application with Docker Compose:

- Next.js container
- FastAPI container
- PostgreSQL container
- vLLM / llama.cpp container

This mirrors the production deployment while keeping development manageable.

---

# Concepts Learned

By the end of Phase 0, you should understand:

- Full-stack architecture
- REST API communication
- Model serving
- Local inference
- Docker fundamentals
- Multi-container applications
- Docker Compose
- Environment variables
- Basic deployment workflow
- CI/CD fundamentals

---

# Resume Narrative

> Built and deployed a containerized multi-agent LLM platform using Next.js, FastAPI, PostgreSQL, and a self-hosted lightweight language model, establishing the deployment and inference infrastructure for subsequent ML engineering and MLOps development.
