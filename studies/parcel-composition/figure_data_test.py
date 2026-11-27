"""Exercise the saved-table replot separately from the simulation environment."""

import os
import subprocess
import sys
from pathlib import Path


def test_replot_without_native_splitter(tmp_path):
    """The documented figure command works before a campaign builds tree.so."""
    study = Path(__file__).resolve().parent
    output = tmp_path / "figures"
    env = {
        **os.environ,
        "PYTHONPATH": ".",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        "BOREAL_TREE": str(tmp_path / "unbuilt" / "tree.so"),
    }
    subprocess.run(  # noqa: S603 - fixed study script and interpreter
        [sys.executable, study / "figure_data.py", "--output", output],
        cwd=study.parents[1],
        env=env,
        check=True,
    )
    assert (output / "parcel-redraw-reliability.png").is_file()
