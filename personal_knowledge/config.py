"""Configuration for embedded Release-scoped personal-knowledge retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, model_validator


load_dotenv(override=False)


def knowledge_root() -> Path:
    configured = os.environ.get("PERSONAL_KNOWLEDGE_ROOT", "personal_knowledge_data")
    return Path(configured).expanduser().resolve()


class RetrievalCfg(BaseModel):
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    index_dir: str = ".local/indexes/specifications"
    specification_root: str = "data/3gpp"
    version_policy: Literal[
        "latest_per_spec",
        "highest_version_within_release",
    ] = "highest_version_within_release"
    target_release: int | None = 17
    chunk_characters: int = 2400
    chunk_overlap_characters: int = 300
    semantic_candidates_per_collection: int = 20
    lexical_candidates_per_collection: int = 20
    final_hits_per_collection: int = 6
    rrf_k: int = 60

    @model_validator(mode="after")
    def _valid_retrieval_mechanics(self):
        if (
            self.version_policy == "highest_version_within_release"
            and self.target_release is None
        ):
            raise ValueError("target_release is required")
        if self.version_policy == "latest_per_spec" and self.target_release is not None:
            raise ValueError("target_release is invalid for latest_per_spec")
        if not 0 <= self.chunk_overlap_characters < self.chunk_characters:
            raise ValueError("chunk overlap must be below chunk size")
        return self


class AppConfig(BaseModel):
    retrieval: RetrievalCfg = RetrievalCfg()


def load_config() -> AppConfig:
    release = int(os.environ.get("PERSONAL_KNOWLEDGE_RELEASE", "17"))
    return AppConfig(
        retrieval=RetrievalCfg(
            embedding_model=os.environ.get(
                "PERSONAL_KNOWLEDGE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            index_dir=os.environ.get(
                "PERSONAL_KNOWLEDGE_INDEX_DIR", ".local/indexes/specifications"
            ),
            specification_root=os.environ.get(
                "PERSONAL_KNOWLEDGE_CORPUS_DIR", "data/3gpp"
            ),
            target_release=release,
        )
    )
