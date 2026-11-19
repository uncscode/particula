"""Save trajectory refinements and fresh-process GPU allocation measurements."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scripts.inverse_studies.coagulation_reference import (
    assess as compare_clusters,
)
from scripts.inverse_studies.forward_checks import assess


def qualify(output):
    """Run both trajectory backends and compare full and segmented tapes."""
    rows = assess("cpu") + assess("cuda:0")
    measurements = []
    with tempfile.TemporaryDirectory() as scratch:
        cell = Path(scratch) / "cell.json"
        for boxes in (2, 128):
            for steps in (32, 128):
                for segment in (0, 8):
                    subprocess.run(  # noqa: S603 - fixed module and numeric grid
                        [
                            sys.executable,
                            "-m",
                            "scripts.inverse_studies.tape_memory",
                            str(boxes),
                            str(steps),
                            str(segment),
                            str(cell),
                        ],
                        check=True,
                    )
                    measurements.append(json.loads(cell.read_text()))
                full, segmented = measurements[-2:]
                for field in ("radius_um", "adjoint"):
                    np.testing.assert_allclose(
                        full[field], segmented[field], rtol=1e-12
                    )
    sources = [
        Path("particula/execution/inverse_condensation.py"),
        *sorted(Path("scripts/inverse_studies").glob("*.py")),
    ]
    report = dict(
        coagulation=compare_clusters(),
        refinement=rows,
        memory=measurements,
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    qualify(parser.parse_args().output)
