"""Runtime smoke coverage for the selected coagulation example."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.warp

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/Examples/gpu_coagulation_direct.py"
)


def test_example_forced_no_warp_path() -> None:
    """The example exits successfully when Warp is explicitly disabled."""
    process = subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
        timeout=10,
    )
    assert process.returncode == 0
