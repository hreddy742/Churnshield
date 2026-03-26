from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    app_name: str = "ChurnShield API"
    debug: bool = False
    database_url: str = "postgresql://localhost/churnshield"
    models_dir: Path = Path("ml/models")
    data_dir: Path = Path("ml/data/raw")
    cors_origins: list = ["http://localhost:3000", "https://churnshield.vercel.app"]

    class Config:
        env_file = ".env"

settings = Settings()
