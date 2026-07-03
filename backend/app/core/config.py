from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LLMposter API"
    app_env: str = Field(default="development", alias="APP_ENV")
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/llmposter",
        alias="DATABASE_URL",
    )

    model_server_url: str = Field(default="http://localhost:11434", alias="MODEL_SERVER_URL")
    model_name: str = Field(default="qwen2.5:1.5b", alias="MODEL_NAME")
    inference_mode: str = Field(default="remote", alias="INFERENCE_MODE")
    agent_config_source: str = Field(default="static", alias="AGENT_CONFIG_SOURCE")
    word_selection_mode: str = Field(default="random", alias="WORD_SELECTION_MODE")
    fixed_secret_word: str = Field(default="satellite", alias="FIXED_SECRET_WORD")
    fixed_imposter_hint: str = Field(
        default="Space, signals, or orbit",
        alias="FIXED_IMPOSTER_HINT",
    )
    backend_cors_origins: str = Field(
        default="http://localhost:3000",
        alias="BACKEND_CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.backend_cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
