"""Run the Mt. Soledad study in a separate output tree, preserving the published files."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parent
REPO = SOURCE.parents[2]
MINUTE_SHA = "8e779e92c897f34b4a2ccc7677a4fa99ddb5d018c1597ec6fbf66bcbd6ba3bf8"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=["check", "extract", "fit-original", "field", "film"]
    )
    p.add_argument("output", type=Path, help="separate scratch/output directory")
    p.add_argument("--inputs", type=Path, help="local downloads, named as in README.md")
    a = p.parse_args()
    out = a.output.resolve()
    if SOURCE == out or SOURCE in out.parents or out in SOURCE.parents:
        p.error("Choose an output directory separate from the published source")
    model = out / "model"
    raw = (a.inputs or (out / "inputs")).resolve()
    if model == raw or model in raw.parents:
        p.error("Keep source downloads outside the generated model directory")
    model.mkdir(parents=True, exist_ok=True)
    # Preserve regenerated inputs when resuming another stage.
    for path in SOURCE.rglob("*"):
        if not path.is_file() or any(
            v.startswith(".") or v == "__pycache__"
            for v in path.relative_to(SOURCE).parts
        ):
            continue
        if path.suffix not in {".py", ".csv", ".npz"}:
            continue
        dest = model / path.relative_to(SOURCE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".py" or not dest.exists():
            shutil.copyfile(path, dest)
    ext = model / "extension"
    chem = model / "field-mixtures/chem"
    cmp = chem.parent / "comparison"
    obs = chem.parent / "observations"
    films = chem.parent / "films"
    drops = model / "mechanisms/droplets"
    env = os.environ.copy()
    repo = Path(env.get("PARTICULA_ROOT", REPO)).resolve()
    base_paths = [
        model,
        ext,
        chem,
        cmp,
        drops,
        model / "organic-films/kinetic",
        repo,
        repo / "packages/particula-aiomfac/src",
    ]
    env.update(
        SOLEDAD_INPUT=str(raw),
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR=str(out / "mpl-cache"),
    )

    def run(script, *args, cwd=None):
        path = model / script
        # Each file's own directory comes first, e.g. extension/edges vs films/edges.
        env["PYTHONPATH"] = os.pathsep.join(str(x) for x in base_paths)
        subprocess.run(
            [sys.executable, str(path), *map(str, args)],
            cwd=cwd or path.parent,
            env=env,
            check=True,
        )

    if a.stage == "check":
        run("check.py")
    elif a.stage == "extract":
        processed = raw / "ccn-processed.zip"
        logger = raw / "ccn-raw.zip"
        run("extension/detector.py", processed, ext / "blocks-final.npz")
        run("extension/edges.py", ext / "blocks-final.npz", ext / "edges.csv")
        run(
            "extension/instrument.py", ext / "blocks-final.npz", ext / "diagnostics.csv"
        )
        run("extension/acquisition.py", logger, ext / "blocks-final.npz", ext)
        run("mechanisms/droplets/extract.py", processed, logger, ext, drops)
        mass = ext / "chemistry"
        run(
            "extension/chemistry/extract_ams.py",
            raw / "ams.nc",
            raw / "ams-kappa.nc",
            processed,
            mass,
        )
        run("extension/chemistry/extract_cims.py", raw / "gas.zip", mass)
        run("organic-films/bulk/composition.py")
        minute = raw / "minute-flow.csv"
        if not minute.exists():
            minute = out / "minute-flow.csv"
            run(
                "field-mixtures/comparison/extract_flow_member.py",
                raw / "filter-flow.zip",
                minute,
            )
        with minute.open("rb") as input_file:
            if hashlib.file_digest(input_file, "sha256").hexdigest() != MINUTE_SHA:
                raise ValueError("Unexpected minute-series checksum")
        run("field-mixtures/comparison/filter_flow.py", minute)
        run("field-mixtures/chem/recipe.py")
        run("field-mixtures/chem/c16.py")
        run(
            "field-mixtures/observations/observations.py",
            "--growth",
            raw / "growth.zip",
            "--filters",
            raw / "ftir.csv",
        )
        shutil.copyfile(raw / "htdma-summary.png", obs / "author-summary.png")
        run("field-mixtures/observations/calibration.py")
        run("field-mixtures/observations/clock_profile.py")
        run("mechanisms/droplets/models.py")
    elif a.stage == "fit-original":
        run("extract.py", raw / "ccn-processed.zip", model / "data")
        run("study.py", model / "data", model / "final")
    elif a.stage == "field":
        run(
            "mechanisms/mixing/mixing.py",
            "--ext",
            ext,
            "--out",
            model / "mechanisms/mixing/final",
            "--grid",
            513,
            "--count",
            128,
            "--draws",
            5,
        )
        run("field-mixtures/chem/thermo.py", "central", "cuda")
        run("field-mixtures/chem/thermo.py", "refine", "cuda")
        run("field-mixtures/chem/thermo.py", "uncertainty", "cuda")
        run("field-mixtures/chem/hydration.py")
        run("field-mixtures/observations/partial_uptake.py")
        run("field-mixtures/comparison/response.py")
        run("field-mixtures/comparison/field_score.py")
    else:
        run("field-mixtures/films/prepare.py", "anchored", films / "inputs.csv")
        run(
            "field-mixtures/films/outlet.py",
            films / "inputs.csv",
            films / "final",
            "--steps",
            513,
            "--detail",
        )
        run(
            "field-mixtures/films/edges.py",
            films / "inputs.csv",
            films / "final",
            "--grid",
            1025,
        )


if __name__ == "__main__":
    main()
