"""Small structured-model boundary used by local research components."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel


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

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def complete_json(self, prompt: str) -> dict[str, Any]:
        response = self._model.invoke(prompt)
        return _parse_json_object(_message_text(response.content))


__all__ = [
    "LangChainStructuredResearchModel",
    "StructuredResearchModel",
]
