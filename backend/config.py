"""Validated runtime configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application-owned settings.

    Provider-specific GPT Researcher variables remain in the environment and
    are intentionally not duplicated here.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ENV_FILE,
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
    personal_knowledge_model_max_output_tokens: int = Field(
        default=4096, ge=512, le=32768
    )
    personal_knowledge_max_tasks: int = Field(default=3, ge=1, le=3)
    personal_knowledge_max_retrieval_rounds: int = Field(default=4, ge=1, le=12)
    personal_knowledge_max_parallel_tasks: int = Field(default=3, ge=1, le=3)
    personal_knowledge_max_tool_calls_per_task: int = Field(default=12, ge=1, le=24)

    def resolved_outputs_dir(self) -> Path:
        path = self.outputs_dir.expanduser()
        return (
            (PROJECT_ROOT / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )

    def resolved_personal_knowledge_root(self) -> Path:
        path = self.personal_knowledge_root.expanduser()
        return (
            (PROJECT_ROOT / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )

    def resolved_personal_knowledge_path(self, configured: str, label: str) -> Path:
        root = self.resolved_personal_knowledge_root()
        path = (root / configured).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{label} escapes Personal Knowledge root: {configured}")
        return path

    def validate_for_personal_research(self) -> None:
        corpus = self.resolved_personal_knowledge_path(
            self.personal_knowledge_corpus_dir, "Personal knowledge corpus"
        )
        index = self.resolved_personal_knowledge_path(
            self.personal_knowledge_index_dir, "Personal knowledge index"
        )
        if not corpus.is_dir():
            raise ValueError(f"Personal knowledge corpus does not exist: {corpus}")
        if not index.is_dir() and not self.personal_knowledge_allow_index_build:
            raise ValueError(f"Personal knowledge index does not exist: {index}")


def load_settings() -> Settings:
    # GPT Researcher and model providers read their own configuration directly
    # from os.environ. Make the repository-local file authoritative so stale
    # credentials inherited from the launching shell cannot silently replace
    # the key the user configured for this project.
    load_dotenv(dotenv_path=PROJECT_ENV_FILE, override=True)
    return Settings()
