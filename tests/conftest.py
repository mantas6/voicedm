import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run each test in its own directory, so the default ``./data`` database is fresh."""
    monkeypatch.chdir(tmp_path)
