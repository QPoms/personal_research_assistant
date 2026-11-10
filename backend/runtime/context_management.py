"""Deep Agents context compaction configured for research workloads."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
)
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel

from backend.config import Settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextCompactionNotice:
    agent: str
    messages_before: int
    approximate_tokens_before: int
    cutoff_index: int
    history_path: str | None
    trigger_tokens: int
    keep_messages: int

    def payload(self) -> dict[str, str | int | None]:
        return {
            "messages_before": self.messages_before,
            "approximate_tokens_before": self.approximate_tokens_before,
            "cutoff_index": self.cutoff_index,
            "history_path": self.history_path,
            "trigger_tokens": self.trigger_tokens,
            "keep_messages": self.keep_messages,
        }


class ObservableSummarizationMiddleware(SummarizationMiddleware):
    """Use Deep Agents compaction while emitting a small application event."""

    @property
    def name(self) -> str:
        # Deep Agents replaces core middleware by this public middleware name.
        return "SummarizationMiddleware"

    def __init__(
        self,
        model: BaseChatModel,
        *,
        backend: BackendProtocol,
        agent: str,
        trigger_tokens: int,
        keep_messages: int,
        summary_input_tokens: int,
        on_compaction: Callable[[ContextCompactionNotice], None] | None = None,
    ) -> None:
        super().__init__(
            model,
            backend=backend,
            trigger=("tokens", trigger_tokens),
            keep=("messages", keep_messages),
            trim_tokens_to_summarize=summary_input_tokens,
            truncate_args_settings={
                "trigger": ("tokens", max(1, trigger_tokens // 2)),
                "keep": ("messages", keep_messages),
                "max_length": 2000,
                "truncation_text": "...[argument truncated; original content is in an artifact]",
            },
        )
        self._application_agent = agent
        self._application_trigger_tokens = trigger_tokens
        self._application_keep_messages = keep_messages
        self._on_compaction = on_compaction

    def _metrics(self, request: ModelRequest) -> tuple[int, int]:
        messages = self._get_effective_messages(request)
        tokens = self._count_tokens(messages, request.system_message, request.tools)
        return len(messages), tokens

    def _notify(
        self,
        response: ModelResponse | ExtendedModelResponse,
        metrics: tuple[int, int],
    ) -> None:
        if self._on_compaction is None:
            return
        command = getattr(response, "command", None)
        update = getattr(command, "update", None)
        if not isinstance(update, Mapping):
            return
        event = update.get("_summarization_event")
        if not isinstance(event, Mapping):
            return
        cutoff = event.get("cutoff_index")
        if not isinstance(cutoff, int) or cutoff <= 0:
            return
        history_path = event.get("file_path")
        try:
            self._on_compaction(
                ContextCompactionNotice(
                    agent=self._application_agent,
                    messages_before=metrics[0],
                    approximate_tokens_before=metrics[1],
                    cutoff_index=cutoff,
                    history_path=(
                        history_path if isinstance(history_path, str) else None
                    ),
                    trigger_tokens=self._application_trigger_tokens,
                    keep_messages=self._application_keep_messages,
                )
            )
        except Exception:
            logger.exception("Context compaction event callback failed")

    def wrap_model_call(self, request: ModelRequest, handler: Callable[..., Any]):
        metrics = self._metrics(request)
        response = super().wrap_model_call(request, handler)
        self._notify(response, metrics)
        return response

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[..., Any]
    ):
        metrics = self._metrics(request)
        response = await super().awrap_model_call(request, handler)
        self._notify(response, metrics)
        return response


def build_context_middleware_stack(
    settings: Settings,
    model: BaseChatModel,
    backend: BackendProtocol,
    *,
    agent: str,
    on_compaction: Callable[[ContextCompactionNotice], None] | None = None,
) -> list[Any]:
    summarization = ObservableSummarizationMiddleware(
        model,
        backend=backend,
        agent=agent,
        trigger_tokens=settings.context_summarization_trigger_tokens,
        keep_messages=settings.context_summarization_keep_messages,
        summary_input_tokens=settings.context_summary_input_tokens,
        on_compaction=on_compaction,
    )
    return [
        summarization,
        SummarizationToolMiddleware(summarization, system_prompt=None),
    ]


__all__ = [
    "ContextCompactionNotice",
    "ObservableSummarizationMiddleware",
    "build_context_middleware_stack",
]
