"""Small structured-model boundary used by local research components."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from backend.runtime.model_retry import ModelRetryPolicy, is_retryable_model_error


class StructuredResearchModel(Protocol):
    def complete_json(self, prompt: str) -> dict[str, Any]: ...


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content or "")


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response did not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


class LangChainStructuredResearchModel:
    """Use one preconfigured chat model for bounded JSON-only decisions."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        retry_policy: ModelRetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model
        self._retry_policy = retry_policy or ModelRetryPolicy()
        self._sleep = sleep

    def complete_json(self, prompt: str) -> dict[str, Any]:
        last_error: Exception | None = None
        diagnostics: list[str] = []
        attempts = self._retry_policy.max_retries + 1
        for attempt in range(attempts):
            request = prompt
            if attempt:
                request += (
                    "\n\nYour previous response was not a parseable JSON object. "
                    "Return the requested JSON object now, with no analysis, "
                    "Markdown, or surrounding prose."
                )
            try:
                response = self._model.invoke(request)
            except Exception as exc:
                if (
                    not is_retryable_model_error(exc)
                    or attempt >= self._retry_policy.max_retries
                ):
                    raise
                self._sleep(self._retry_policy.delay_seconds(attempt))
                continue
            candidates = [("content", _message_text(response.content))]
            reasoning = response.additional_kwargs.get("reasoning_content")
            if isinstance(reasoning, str):
                candidates.append(("reasoning_content", reasoning))
            if not any(text.strip() for _, text in candidates):
                diagnostics.append(f"attempt {attempt + 1}: empty response")
            for source, text in candidates:
                if not text.strip():
                    continue
                try:
                    return _parse_json_object(text)
                except (ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    preview = text.strip().replace("\n", " ")[:240]
                    diagnostics.append(f"attempt {attempt + 1} {source}: {preview!r}")
            if attempt < self._retry_policy.max_retries:
                self._sleep(self._retry_policy.delay_seconds(attempt))
        detail = "; ".join(diagnostics) or "no response diagnostics"
        raise ValueError(
            "model response did not contain a JSON object; " + detail
        ) from last_error


__all__ = [
    "LangChainStructuredResearchModel",
    "StructuredResearchModel",
]
