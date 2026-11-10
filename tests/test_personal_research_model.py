from types import SimpleNamespace

from personal_knowledge.research.model import LangChainStructuredResearchModel


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
    model = LangChainStructuredResearchModel(inner)

    assert model.complete_json("test") == {"status": "partial"}
    assert inner.calls == 2


class AlwaysMalformedModel:
    def invoke(self, prompt):
        return SimpleNamespace(content="plain prose", additional_kwargs={})


def test_structured_model_failure_includes_bounded_response_diagnostic():
    model = LangChainStructuredResearchModel(AlwaysMalformedModel())

    try:
        model.complete_json("test")
    except ValueError as exc:
        assert "attempt 1 content: 'plain prose'" in str(exc)
        assert "attempt 2 content: 'plain prose'" in str(exc)
    else:
        raise AssertionError("malformed responses were accepted")
