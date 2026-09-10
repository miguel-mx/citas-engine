"""Configuration — env-driven, mirrors Cell 3 of the notebook.

Values are read once at import into a `settings` singleton. Only OpenAlex is
required to run; the other sources are optional and enabled when a key is set.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── OpenAlex (always on) ──────────────────────────────────────────────────
    # mailto keeps us in OpenAlex's "polite pool"; use a real CCM contact.
    mailto: str = "itprojects0@gmail.com"
    openalex_base: str = "https://api.openalex.org"

    # ── Ollama (report generation only) ───────────────────────────────────────
    ollama_base: str = "http://localhost:11434"
    model: str = "gemma4:e4b"
    # A local model writing a full report can take minutes on CPU; this ceiling is
    # generous on purpose, and only /report and /analyze?want_report=true wait on it.
    ollama_timeout: float = 300.0

    # ── Fetch limits / tuning ─────────────────────────────────────────────────
    max_works: int = 500
    max_citing_per_work: int = 1000
    count_tolerance: float = 0.10
    page_size: int = 200
    max_workers: int = 5
    max_retry_after_seconds: int = 30

    # ── Optional external sources (plug in later) ─────────────────────────────
    scopus_api_key: str = ""
    wos_api_key: str = ""

    @property
    def use_scopus(self) -> bool:
        return bool(self.scopus_api_key)

    @property
    def use_wos(self) -> bool:
        return bool(self.wos_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
