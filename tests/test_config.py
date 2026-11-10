import backend.config as config
from backend.config import Settings


def test_load_settings_makes_project_env_authoritative(monkeypatch):
    calls = []
    monkeypatch.setattr(
        config,
        "load_dotenv",
        lambda **kwargs: calls.append(kwargs),
    )

    config.load_settings()

    assert calls == [{"dotenv_path": config.PROJECT_ENV_FILE, "override": True}]


def test_relative_runtime_paths_are_anchored_to_project(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    settings = Settings(
        outputs_dir="outputs",
        personal_knowledge_root="knowledge",
    )

    assert settings.resolved_outputs_dir() == tmp_path / "outputs"
    assert settings.resolved_personal_knowledge_root() == tmp_path / "knowledge"


def test_personal_subpaths_cannot_escape_root(tmp_path):
    settings = Settings(personal_knowledge_root=tmp_path)

    try:
        settings.resolved_personal_knowledge_path("../escape", "corpus")
    except ValueError as exc:
        assert "escapes Personal Knowledge root" in str(exc)
    else:
        raise AssertionError("escaping Personal Knowledge path was accepted")
