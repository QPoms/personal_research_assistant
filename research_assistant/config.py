"""Validated runtime configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


class Settings(BaseSettings):
    """Application-owned settings.

    Provider-specific GPT Researcher variables remain in the environment and
    are intentionally not duplicated here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    deep_agent_model: str = "openai:gpt-5.4"
    outputs_dir: Path = Path("outputs")
    max_parallel_researchers: int = Field(default=3, ge=1, le=16)
    max_revision_rounds: int = Field(default=1, ge=0, le=3)
    max_search_results: int = Field(default=5, ge=1, le=20)
    research_tool_timeout_seconds: float = Field(default=900.0, gt=0)

    specrag_base_url: str = "http://127.0.0.1:8000"
    specrag_workspace_id: str | None = None
    specrag_access_token: SecretStr | None = None
    specrag_release: int = Field(default=17, ge=1)
    specrag_answer_language: str = "Chinese"
    specrag_detail_level: str = "standard"
    specrag_poll_interval_seconds: float = Field(default=2.0, gt=0)
    specrag_poll_timeout_seconds: float = Field(default=60.0, ge=0)
    specrag_http_timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("specrag_base_url")
    @classmethod
    def strip_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    def validate_for_personal_research(self) -> None:
        missing: list[str] = []
        if not self.specrag_workspace_id:
            missing.append("SPECRAG_WORKSPACE_ID")
        if not self.specrag_access_token:
            missing.append("SPECRAG_ACCESS_TOKEN")
        if missing:
            raise ValueError(
                "Personal research requires configured SpecRAG access: "
                + ", ".join(missing)
            )


def load_settings() -> Settings:
    # GPT Researcher and model providers read their own configuration directly
    # from os.environ. Load the shared file before validating app-owned fields.
    load_dotenv(override=False)
    return Settings()


