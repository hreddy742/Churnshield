"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:  # pragma: no cover
    from pydantic import BaseModel

    class BaseSettings(BaseModel):  # type: ignore[misc]
        """Fallback settings base when pydantic-settings is unavailable."""

        model_config = {"arbitrary_types_allowed": True}

        def __init__(self, **data):
            import os

            merged = {**{key: os.getenv(key) for key in self.__class__.model_fields}, **data}
            super().__init__(**merged)

    def SettingsConfigDict(**kwargs):  # type: ignore[no-redef]
        return kwargs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings for ChurnGuard."""

    DATABASE_URL: str = "sqlite:///./data/churnguard.db"
    MLFLOW_TRACKING_URI: str = "./mlruns"
    MODEL_PATH: str = str(PROJECT_ROOT / "models" / "ensemble.pkl")
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton application settings."""

    return Settings()
