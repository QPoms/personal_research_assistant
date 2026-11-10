import pytest

from backend.storage.run_paths import create_run_paths


def test_create_isolated_run_paths(tmp_path):
    paths = create_run_paths(tmp_path, "run-123")
    assert paths.root == tmp_path.resolve() / "run_run-123"
    assert paths.sections.is_dir()
    assert paths.report == paths.root / "report.md"


@pytest.mark.parametrize("unsafe", ["../escape", "a/b", "a\\b", "", " space"])
def test_run_id_cannot_escape_outputs_directory(tmp_path, unsafe):
    with pytest.raises(ValueError):
        create_run_paths(tmp_path, unsafe)
