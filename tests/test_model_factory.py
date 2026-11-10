import backend.runtime.model_factory as model_factory


def test_openai_compatible_model_uses_explicit_protocol(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_init(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(model_factory, "init_chat_model", fake_init)

    result = model_factory.deep_agent_model(
        " openai:deepseek-v4.1-flash ",
        max_output_tokens=2048,
        use_responses_api=False,
    )

    assert result is sentinel
    assert captured == {
        "model_name": "openai:deepseek-v4.1-flash",
        "kwargs": {
            "max_tokens": 2048,
            "max_retries": 0,
            "use_responses_api": False,
        },
    }


def test_non_openai_model_does_not_receive_openai_protocol_flag(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        model_factory,
        "init_chat_model",
        lambda model_name, **kwargs: captured.update(
            model_name=model_name, kwargs=kwargs
        ),
    )

    model_factory.deep_agent_model(
        "anthropic:claude-test",
        max_output_tokens=1024,
        use_responses_api=False,
    )

    assert captured == {
        "model_name": "anthropic:claude-test",
        "kwargs": {"max_tokens": 1024, "max_retries": 0},
    }
