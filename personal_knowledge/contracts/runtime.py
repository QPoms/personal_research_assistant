"""Dependency-free failure vocabulary retained from SpecRAG runtime models."""

from typing import Literal

RunFailureKind = Literal[
    "retryable_transient",
    "rate_limited",
    "timeout",
    "provider_unavailable",
    "authentication_or_configuration",
    "invalid_request",
    "structured_output",
    "model_refusal",
    "context_overflow",
    "internal_error",
    "unknown",
]

RUN_FAILURE_KINDS: frozenset[str] = frozenset({
    "retryable_transient",
    "rate_limited",
    "timeout",
    "provider_unavailable",
    "authentication_or_configuration",
    "invalid_request",
    "structured_output",
    "model_refusal",
    "context_overflow",
    "internal_error",
    "unknown",
})

__all__ = ["RUN_FAILURE_KINDS", "RunFailureKind"]
