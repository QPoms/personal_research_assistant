import backend.config as config


def test_load_settings_makes_project_env_authoritative(monkeypatch):
    calls = []
    monkeypatch.setattr(
        config,
        "load_dotenv",
        lambda **kwargs: calls.append(kwargs),
    )

    config.load_settings()

    assert calls == [{"override": True}]
