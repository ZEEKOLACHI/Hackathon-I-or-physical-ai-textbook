"""Application configuration using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Find .env file - check current dir, then parent (project root)
_env_file = Path(".env")
if not _env_file.exists():
    _env_file = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_env_file) if _env_file.exists() else ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Google AI (Gemini)
    google_api_key: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./local.db"

    # Better-Auth
    better_auth_secret: str = ""
    better_auth_url: str = "http://localhost:3000"

    # Backend
    backend_url: str = "https://physical-ai-backend.vercel.app"
    cors_origins: str = "https://physical-ai-textbook.vercel.app,http://localhost:3000"

    # Development
    debug: bool = False

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
