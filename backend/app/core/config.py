import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    chat_url: str
    model: str
    api_key_env: str | None = None
    min_max_tokens: int = 128
    structured_max_tokens: int = 512
    notes: str | None = None


DEFAULT_LLM_CONFIG = LLMConfig(
    name="OpenRouter default",
    provider="openrouter",
    chat_url="https://openrouter.ai/api/v1/chat/completions",
    model="openrouter/free",
    api_key_env="OPENROUTER_API_KEY",
    min_max_tokens=128,
    structured_max_tokens=512,
)


class Settings(BaseSettings):
    app_name: str = "LLMposter API"
    app_env: str = Field(default="development", alias="APP_ENV")
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/llmposter",
        alias="DATABASE_URL",
    )

    llm_experiment_config: str | None = Field(default=None, alias="LLM_EXPERIMENT_CONFIG")
    embedding_model_server_url: str | None = Field(
        default="http://localhost:11434",
        alias="EMBEDDING_MODEL_SERVER_URL",
    )
    embedding_model_name: str = Field(default="nomic-embed-text", alias="EMBEDDING_MODEL_NAME")
    inference_mode: str = Field(default="remote", alias="INFERENCE_MODE")
    agent_config_source: str = Field(default="static", alias="AGENT_CONFIG_SOURCE")
    word_selection_mode: str = Field(default="random", alias="WORD_SELECTION_MODE")
    fixed_secret_word: str = Field(default="satellite", alias="FIXED_SECRET_WORD")
    fixed_imposter_hint: str = Field(default="orbit", alias="FIXED_IMPOSTER_HINT")
    backend_cors_origins: str = Field(
        default="http://localhost:3000",
        alias="BACKEND_CORS_ORIGINS",
    )
    backend_cors_origin_regex: str | None = Field(
        default=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
        alias="BACKEND_CORS_ORIGIN_REGEX",
    )

    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", BACKEND_DIR / ".env"),
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]

    def load_llm_config(self) -> LLMConfig:
        if not self.llm_experiment_config:
            return DEFAULT_LLM_CONFIG

        config_path = Path(self.llm_experiment_config)
        if not config_path.is_absolute():
            config_path = REPO_DIR / config_path

        with config_path.open("r", encoding="utf-8") as config_file:
            return LLMConfig.model_validate(json.load(config_file))

    def get_env_value(self, name: str) -> str | None:
        value = os.getenv(name)
        if value is not None:
            return value

        for env_file in self.model_config.get("env_file", ()):
            env_value = _read_env_file_value(Path(env_file), name)
            if env_value is not None:
                return env_value

        return None


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


settings = get_settings()
