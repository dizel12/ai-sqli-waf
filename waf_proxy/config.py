from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False)

    upstream_url: str = "http://vuln-app:8000"
    inference_url: str = "http://inference-svc:9000"
    inference_timeout_ms: int = 250
    active_model: str = "cnn"
    block_threshold: float = 0.5
    min_value_len: int = 3
    events_db: str = "/data/events.db"


def get_settings() -> Settings:
    return Settings()
