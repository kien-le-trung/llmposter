"""Local deterministic OpenAI-compatible provider; never contacts a remote model."""

import asyncio
import json
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def decision(prompt: str) -> dict:
    schema, _ = json.JSONDecoder().raw_decode(prompt.split("JSON schema:\n", 1)[1])
    properties = schema.get("properties", {})
    if "clue" in properties:
        return {"clue": "distant signal"}
    context, _ = json.JSONDecoder().raw_decode(prompt.split("GAME_CONTEXT:\n", 1)[1])
    if "target_id" in properties:
        return {"target_id": next(p["id"] for p in context["players"]
                                  if p["id"] != context["your_id"])}
    return {"comment": None if context.get("must_stay_silent") else "How does that clue fit?",
            "memory": {"summary": "", "observations": [], "suspicions": [], "questions": []}}


def create_app(delay: float | None = None, scenario: str | None = None):
    app = FastAPI()
    delay = float(os.environ.get("LATENCY_STUB_DELAY", "0.2")) if delay is None else delay
    scenario = scenario or os.environ.get("LATENCY_STUB_SCENARIO", "normal")
    app.state.calls = 0
    app.state.active = 0
    app.state.peak = 0

    @app.get("/health")
    def health():
        return {"status": "ok", "peak": app.state.peak}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        prompt = body["messages"][-1]["content"]
        app.state.calls += 1
        call = app.state.calls
        app.state.active += 1
        app.state.peak = max(app.state.peak, app.state.active)
        try:
            wait = delay
            if scenario == "timeout" and call == 1:
                wait = 21  # Exceeds the real inference client's 20-second HTTP timeout.
            if scenario == "deadline" and '"comment"' in prompt.split("JSON schema:\n")[-1]:
                wait = 60
            await asyncio.sleep(wait)
            if scenario == "rate_limit" and call == 1:
                return JSONResponse({"error": {"message": "controlled rate limit"}},
                                    status_code=429, headers={"Retry-After": "0.05"})
            value = {} if scenario == "invalid" and call == 1 else decision(prompt)
            if scenario == "silence" and "comment" in value:
                value["comment"] = None
            return {"id": f"stub-{call}", "provider": "local-stub", "model": "latency-stub",
                    "choices": [{"message": {"content": json.dumps(value)},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(prompt) // 4,
                              "completion_tokens": 24, "total_tokens": len(prompt) // 4 + 24}}
        finally:
            app.state.active -= 1

    return app
