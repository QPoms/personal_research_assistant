"""Model construction for Deep Agents.

Deep Agents enables OpenAI's Responses API when it receives a provider-qualified
string. Many OpenAI-compatible gateways implement only Chat Completions, so the
application constructs the model explicitly and makes that protocol choice
configurable. Credentials remain in the environment.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def deep_agent_model(
    model_name: str,
    *,
    max_output_tokens: int,
    use_responses_api: bool,
) -> BaseChatModel:
    value = model_name.strip()
    if not value:
        raise ValueError("DEEP_AGENT_MODEL must not be empty")

    model_kwargs: dict[str, object] = {"max_tokens": max_output_tokens}
    if value.partition(":")[0].lower() == "openai":
        model_kwargs["use_responses_api"] = use_responses_api
    return init_chat_model(value, **model_kwargs)


