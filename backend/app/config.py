import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    app_name: str = "Analytiq"
    data_dir: Path = Path("./data")
    static_dir: Path | None = None
    database_url: str = ""
    jwt_secret: str = ""
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    worker_mode: str = "embedded"
    max_workers: int = 2
    enable_demo: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    max_upload_mb: int = 50
    max_dataset_rows: int = 100_000


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir = settings.data_dir.resolve()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("datasets", "models", "reports"):
        (settings.data_dir / folder).mkdir(exist_ok=True)
    if not settings.database_url:
        settings.database_url = f"sqlite:///{settings.data_dir / 'analytiq.db'}"
    if not settings.jwt_secret:
        secret_path = settings.data_dir / ".jwt-secret"
        try:
            with secret_path.open("x") as file:
                file.write(secrets.token_urlsafe(48))
            try:
                secret_path.chmod(0o600)
            except OSError:
                # Windows-mounted WSL directories may not support Unix mode changes.
                pass
        except FileExistsError:
            pass
        settings.jwt_secret = secret_path.read_text().strip()
    return settings


settings = get_settings()
