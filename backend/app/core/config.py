import os
from functools import lru_cache
from pathlib import Path

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
BACKEND_ENV_FILE = BACKEND_DIR / ".env"


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    chat_url: str
    model: str
    api_key_env: str | None = None
    min_max_tokens: int = 10000
    structured_max_tokens: int = 10000
    max_concurrent_requests: int | None = Field(default=None, ge=1)
    notes: str | None = None


DEFAULT_LLM_CONFIG = LLMConfig(
    name="OpenRouter default",
    provider="openrouter",
    chat_url="https://openrouter.ai/api/v1/chat/completions",
    model="openrouter/free",
    api_key_env="OPENROUTER_API_KEY",
    min_max_tokens=10000,
    structured_max_tokens=10000,
)


class Settings(BaseSettings):
    # Application metadata.
    app_name: str = "LLMposter API"

    # Settings from backend/.env.example.
    app_env: str = Field(default="development", alias="APP_ENV")
    backend_cors_origins: str = Field(
        default="http://localhost:3000",
        alias="BACKEND_CORS_ORIGINS",
    )
    backend_cors_origin_regex: str | None = Field(
        default=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        alias="BACKEND_CORS_ORIGIN_REGEX",
    )
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/llmposter",
        alias="DATABASE_URL",
    )
    embedding_model_server_url: str | None = Field(
        default="http://localhost:11434",
        alias="EMBEDDING_MODEL_SERVER_URL",
    )
    agent_config_source: str = Field(default="static", alias="AGENT_CONFIG_SOURCE")
    ml_voting_model_path: str = Field(
        default=str(
            BACKEND_DIR
            / "app"
            / "services"
            / "voting"
            / "models"
            / "svm_rbf_balanced_20260712T172943Z"
            / "model.joblib"
        ),
        alias="ML_VOTING_MODEL_PATH",
    )

    # Experiment-tunable settings.
    llm_config: LLMConfig = DEFAULT_LLM_CONFIG
    embedding_model_name: str = Field(default="nomic-embed-text")
    inference_mode: str = Field(default="remote")
    clue_prompt_technique: str = Field(default="few_shot")
    word_selection_mode: str = Field(default="random")
    fixed_secret_word: str = Field(default="satellite")
    fixed_imposter_hint: str = Field(default="orbit")

    model_config = SettingsConfigDict(
        env_file=BACKEND_ENV_FILE,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    def get_env_value(self, name: str) -> str | None:
        value = os.getenv(name)
        if value is not None:
            return value

        return _read_env_file_value(BACKEND_ENV_FILE, name)


def _read_env_file_value(env_file: Path, name: str) -> str | None:
    if not env_file.exists():
        return None

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'")

    return None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


settings = get_settings()
