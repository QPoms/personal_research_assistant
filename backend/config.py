"""Validated runtime configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
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
    deep_agent_max_output_tokens: int = Field(default=2048, ge=128, le=32768)
    deep_agent_use_responses_api: bool = False
    outputs_dir: Path = Path("outputs")
    max_parallel_researchers: int = Field(default=3, ge=1, le=16)
    max_revision_rounds: int = Field(default=1, ge=0, le=3)
    max_search_results: int = Field(default=5, ge=1, le=20)
    research_tool_timeout_seconds: float = Field(default=900.0, gt=0)

    personal_knowledge_root: Path = Path("personal_knowledge_data")
    personal_knowledge_corpus_dir: str = "data/3gpp"
    personal_knowledge_index_dir: str = ".local/indexes/specifications"
    personal_knowledge_release: int = Field(default=17, ge=1)
    personal_knowledge_max_results: int = Field(default=6, ge=1, le=20)
    personal_knowledge_allow_index_build: bool = False
    personal_knowledge_research_enabled: bool = True
    personal_knowledge_allow_external_model: bool = False
    personal_knowledge_research_model: str | None = None
    personal_knowledge_max_tasks: int = Field(default=3, ge=1, le=3)
    personal_knowledge_max_retrieval_rounds: int = Field(default=2, ge=1, le=2)
    personal_knowledge_max_parallel_tasks: int = Field(default=3, ge=1, le=3)

    def validate_for_personal_research(self) -> None:
        root = self.personal_knowledge_root.expanduser().resolve()
        corpus = (root / self.personal_knowledge_corpus_dir).resolve()
        index = (root / self.personal_knowledge_index_dir).resolve()
        if not corpus.is_dir():
            raise ValueError(f"Personal knowledge corpus does not exist: {corpus}")
        if not index.is_dir() and not self.personal_knowledge_allow_index_build:
            raise ValueError(f"Personal knowledge index does not exist: {index}")


def load_settings() -> Settings:
    # GPT Researcher and model providers read their own configuration directly
    # from os.environ. Make the repository-local file authoritative so stale
    # credentials inherited from the launching shell cannot silently replace
    # the key the user configured for this project.
    load_dotenv(override=True)
    return Settings()


