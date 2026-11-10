"""Validated long-term memory contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


MemoryKind = Literal["preference", "finding", "decision", "procedure"]
MemoryTag = Annotated[str, Field(min_length=1, max_length=100)]
MemorySourceRef = Annotated[str, Field(min_length=1, max_length=1000)]


class MemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["memory-entry/v1"] = "memory-entry/v1"
    memory_id: str = Field(pattern=r"^mem_[a-f0-9]{24}$")
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=6_000)
    kind: MemoryKind
    confidence: float = Field(ge=0, le=1)
    tags: tuple[MemoryTag, ...] = Field(default=(), max_length=20)
    source_refs: tuple[MemorySourceRef, ...] = Field(default=(), max_length=20)
    source_run_id: str | None = Field(default=None, max_length=128)
    created_at: datetime
    updated_at: datetime


class MemoryRecall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    score: float = Field(ge=0)
    title: str
    kind: MemoryKind
    confidence: float
    excerpt: str
    tags: tuple[str, ...]
    source_refs: tuple[str, ...]
    source_run_id: str | None


__all__ = [
    "MemoryEntry",
    "MemoryKind",
    "MemoryRecall",
    "MemorySourceRef",
    "MemoryTag",
]
