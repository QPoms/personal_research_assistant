"""Model selection for Deep Agents.

Deep Agents accepts provider-qualified model strings. Credentials stay in the
environment and are never injected into prompts or tool parameters.
"""

from __future__ import annotations


def deep_agent_model(model_name: str) -> str:
    value = model_name.strip()
    if not value:
        raise ValueError("DEEP_AGENT_MODEL must not be empty")
    return value


