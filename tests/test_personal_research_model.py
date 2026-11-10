from types import SimpleNamespace

import pytest

from backend.runtime.model_retry import ModelRetryPolicy
from personal_knowledge.research.model import LangChainStructuredResearchModel


TEST_POLICY = ModelRetryPolicy(
    max_retries=5,
    initial_delay_seconds=0,
    max_delay_seconds=0,
    jitter=False,
)


class FakeReasoningModel:
    def invoke(self, prompt):
        return SimpleNamespace(
            content="",
            additional_kwargs={"reasoning_content": '{"status":"sufficient"}'},
        )


def test_structured_model_accepts_deepseek_reasoning_content_fallback():
    model = LangChainStructuredResearchModel(FakeReasoningModel())

    assert model.complete_json("test") == {"status": "sufficient"}


class RetryModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        content = "not json" if self.calls == 1 else '{"status":"partial"}'
        return SimpleNamespace(content=content, additional_kwargs={})


def test_structured_model_retries_one_malformed_response():
    inner = RetryModel()
    model = LangChainStructuredResearchModel(inner, retry_policy=TEST_POLICY)

    assert model.complete_json("test") == {"status": "partial"}
    assert inner.calls == 2


class AlwaysMalformedModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content="plain prose", additional_kwargs={})


def test_structured_model_failure_includes_bounded_response_diagnostic():
    inner = AlwaysMalformedModel()
    model = LangChainStructuredResearchModel(inner, retry_policy=TEST_POLICY)

    with pytest.raises(ValueError) as captured:
        model.complete_json("test")

    assert inner.calls == 6
    assert "attempt 1 content: 'plain prose'" in str(captured.value)
    assert "attempt 6 content: 'plain prose'" in str(captured.value)


class ProviderError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TransientThenValidModel:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError(429)
        return SimpleNamespace(content='{"status":"sufficient"}', additional_kwargs={})


def test_structured_model_retries_five_transient_failures():
    inner = TransientThenValidModel(failures=5)
    delays = []
    policy = ModelRetryPolicy(
        max_retries=5,
        initial_delay_seconds=2,
        backoff_factor=2,
        max_delay_seconds=30,
        jitter=False,
    )
    model = LangChainStructuredResearchModel(
        inner, retry_policy=policy, sleep=delays.append
    )

    assert model.complete_json("test") == {"status": "sufficient"}
    assert inner.calls == 6
    assert delays == [2, 4, 8, 16, 30]


def test_structured_model_does_not_retry_permanent_provider_error():
    inner = TransientThenValidModel(failures=1)
    inner.invoke = lambda prompt: (_ for _ in ()).throw(ProviderError(403))
    model = LangChainStructuredResearchModel(inner, retry_policy=TEST_POLICY)

    with pytest.raises(ProviderError, match="403"):
        model.complete_json("test")
