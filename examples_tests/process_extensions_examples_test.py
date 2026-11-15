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


def test_gpu_extension_zip_runs_outside_checkout(tmp_path, monkeypatch):
    """The published ZIP carries its Warp helper and runs on real CUDA."""
    import json
    import zipfile

    import pytest

    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from scripts import docs_site

    config = json.loads((ROOT / "website/examples.json").read_text())[
        "custom-process-backends"
    ]
    monkeypatch.setattr(docs_site, "SITE", tmp_path / "site")
    source = (ROOT / config["source"]).read_text()
    docs_site.example_downloads("custom-process-backends", config, source)
    destination = tmp_path / "extracted"
    with zipfile.ZipFile(
        tmp_path / "site/public/downloads/custom-process-backends.zip"
    ) as archive:
        archive.extractall(destination)
    environment = os.environ.copy()
    environment.update(
        MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "matplotlib")
    )
    result = subprocess.run(  # noqa: S603 -- fixed interpreter and local bundle
        [
            sys.executable,
            "-I",
            "-Werror",
            "-c",
            "import sys, runpy; from pathlib import Path; "
            "sys.path.insert(0, str(Path.cwd())); "
            "import matplotlib.pyplot as plt; plt.show = lambda: None; "
            "runpy.run_path('example.py', run_name='__main__')",
        ],
        cwd=destination,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert "NumPy/Warp parity" in result.stdout
