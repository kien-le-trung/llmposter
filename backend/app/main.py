from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import admin, agents, health, rounds
from app.core.config import Settings, get_settings
from app.services.metrics import finish_request_trace, start_request_trace


def create_app(app_settings: Settings | None = None) -> FastAPI:
    active_settings = app_settings if app_settings is not None else get_settings()
    
    app = FastAPI(title=active_settings.app_name)
    app.state.settings = active_settings

    @app.middleware("http")
    async def record_request_metrics(request, call_next):
        trace, token = start_request_trace(request.method, request.url.path)
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            finish_request_trace(trace, token, status_code)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_origin_regex=active_settings.backend_cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(agents.router)
    app.include_router(rounds.router)
    app.include_router(admin.router)

    return app


app = create_app()
