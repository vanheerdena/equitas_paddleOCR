"""Configuration management for the PaddleOCR FastAPI service."""

from functools import lru_cache
from typing import Any

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    api_key: str = Field(..., env="API_KEY", description="API key required for requests.")
    ocr_lang: str = Field(
        "en", env="OCR_LANG", description="Language setting passed to PaddleOCR."
    )
    max_image_bytes: int = Field(
        10 * 1024 * 1024,
        env="MAX_IMAGE_BYTES",
        description="Maximum accepted image size in bytes.",
    )
    request_timeout_seconds: float = Field(
        10.0,
        env="REQUEST_TIMEOUT_SECONDS",
        description="Timeout for downloading remote images.",
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


def reload_settings(**overrides: Any) -> Settings:
    """Reload cached settings, primarily for testing."""

    get_settings.cache_clear()
    return Settings(**overrides)

