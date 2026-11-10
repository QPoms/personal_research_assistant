"""Shared model retry policy for Deep Agents and embedded local research."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import httpx
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.exceptions import ModelError


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable_model_error(exc: Exception) -> bool:
    """Retry only transient provider, transport, and server-side failures."""

    if isinstance(exc, ModelError):
        return exc.is_retryable

    status_code = _status_code(exc)
    if status_code is not None:
        return status_code in {408, 409, 429} or status_code >= 500

    return isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            httpx.NetworkError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
        ),
    )


@dataclass(frozen=True, slots=True)
class ModelRetryPolicy:
    """One explicit retry budget shared by all application model calls."""

    max_retries: int = 5
    initial_delay_seconds: float = 2.0
    backoff_factor: float = 2.0
    max_delay_seconds: float = 30.0
    jitter: bool = True

    def delay_seconds(self, retry_number: int) -> float:
        delay = min(
            self.initial_delay_seconds * (self.backoff_factor**retry_number),
            self.max_delay_seconds,
        )
        if self.jitter and delay > 0:
            delay += random.uniform(-delay * 0.25, delay * 0.25)  # noqa: S311
        return max(0.0, delay)

    def middleware(self) -> ModelRetryMiddleware:
        """Build an independent middleware instance for one agent graph."""

        return ModelRetryMiddleware(
            max_retries=self.max_retries,
            retry_on=is_retryable_model_error,
            on_failure="error",
            backoff_factor=self.backoff_factor,
            initial_delay=self.initial_delay_seconds,
            max_delay=self.max_delay_seconds,
            jitter=self.jitter,
        )


def retry_policy_from_settings(settings: Any) -> ModelRetryPolicy:
    return ModelRetryPolicy(
        max_retries=settings.model_max_retries,
        initial_delay_seconds=settings.model_retry_initial_delay_seconds,
        backoff_factor=settings.model_retry_backoff_factor,
        max_delay_seconds=settings.model_retry_max_delay_seconds,
        jitter=settings.model_retry_jitter,
    )


__all__ = [
    "ModelRetryPolicy",
    "is_retryable_model_error",
    "retry_policy_from_settings",
]
