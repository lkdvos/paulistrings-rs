import re
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def test_import():
    import paulistrings  # noqa: F401


@pytest.mark.skipif(not PYPROJECT.is_file(), reason="pyproject.toml not present (installed wheel, not a repo checkout)")
def test_version_is_the_pyproject_version():
    import paulistrings

    want = re.search(r'^version = "([^"]+)"', PYPROJECT.read_text(), re.MULTILINE).group(1)
    assert paulistrings.__version__ == want
