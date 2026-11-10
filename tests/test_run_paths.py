import pytest

from backend.storage.run_paths import create_run_paths, load_run_paths


def test_create_isolated_run_paths(tmp_path):
    paths = create_run_paths(tmp_path, "run-123")
    assert paths.root == tmp_path.resolve() / "run_run-123"
    assert paths.sections.is_dir()
    assert paths.report == paths.root / "report.md"
    assert paths.events == paths.root / "events.jsonl"


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", "a\\b", "", " space"])
def test_run_id_cannot_escape_outputs_directory(tmp_path, unsafe):
    with pytest.raises(ValueError):
        create_run_paths(tmp_path, unsafe)


def test_load_existing_run_paths_without_recreating_directory(tmp_path):
    created = create_run_paths(tmp_path, "existing")

    loaded = load_run_paths(tmp_path, "existing")

    assert loaded == created


def test_load_run_paths_requires_existing_sections_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="unavailable"):
        load_run_paths(tmp_path, "missing")
