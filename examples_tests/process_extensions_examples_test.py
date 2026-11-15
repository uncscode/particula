"""Public extension tutorials execute independently of checkout helper paths."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_timed_chemistry_download_runs_outside_checkout(tmp_path):
    """The complete downloadable script checks both boxes analytically."""
    source = ROOT / (
        "docs/examples/processes/customization/timed_box_chemistry.py"
    )
    shutil.copy2(source, tmp_path / "example.py")
    environment = os.environ.copy()
    environment.update(
        MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "matplotlib")
    )
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and local example
        [
            sys.executable,
            "-I",
            "-Werror",
            "-c",
            "import runpy; import matplotlib.pyplot as plt; "
            "plt.show = lambda: None; "
            "runpy.run_path('example.py', run_name='__main__')",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "Analytical species and inventory checks passed" in result.stdout
