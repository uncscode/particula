"""Reproduce the Liu evaporation study from digitized inputs in a fresh directory."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

SOURCE = Path(__file__).resolve().parent
CHECKOUT = SOURCE.parents[1]
INPUTS = (
    "sources/observations.csv",
    "sources/fractions.csv",
    "sources/time_verification.csv",
    "sources/peak_check.csv",
    "uncertainty/peaks.csv",
    "viscosity/figure7/figure7_viscosity.csv",
)
STAGES = ("inputs", "chamber", "mechanisms", "viscosity", "cloud")
SOURCE_SUFFIXES = {".py", ".cpp", ".cu", ".sh"}


def prepare(folder, archive):
    """Stage a complete set of inputs and study scripts before exposing the run."""
    if folder.exists():
        raise FileExistsError(
            f"Use a new output directory for inputs or all: {folder}"
        )
    if folder.is_relative_to(SOURCE):
        raise ValueError("Choose a run directory outside the study source tree")
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as temp:
        staged = Path(temp) / "study"
        staged.mkdir()
        for src in SOURCE.rglob("*"):
            if (
                src.is_file()
                and src.suffix in SOURCE_SUFFIXES
                and src.name != "replay.py"
            ):
                target = staged / src.relative_to(SOURCE)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
        with ZipFile(archive) as data:
            for name in INPUTS:
                target = staged / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data.read(name))
        staged.rename(folder)


def reproduce(stage, folder, archive):
    env = dict(os.environ)
    env.update(
        PYTHON=sys.executable,
        PYTHONPATH=str(CHECKOUT),
        PARTICULA_ROOT=str(CHECKOUT),
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR=str(folder / ".matplotlib"),
    )

    def command(*args):
        print("+", " ".join(map(str, args)), flush=True)
        subprocess.run(args, check=True, cwd=folder, env=env)

    def python(script, *args):
        command(sys.executable, script, *args)

    if stage in ("all", "inputs"):
        prepare(folder, archive)
        python("uncertainty/estimate.py", "--out", "uncertainty/results")
    elif not (folder / "uncertainty/results/read_bounds.csv").is_file():
        raise FileNotFoundError(
            "Run the inputs stage at this destination first"
        )

    if stage in ("all", "chamber"):
        for name in (
            "chamber",
            "finite_transfer",
            "mechanisms",
            "published",
            "density_check",
            "figures",
        ):
            python(f"model/{name}.py", "--out", "model/results")
        for name in ("clock_test", "retained_pool"):
            python(f"checks/{name}.py", "--output", "checks")

    if stage in ("all", "mechanisms"):
        for name in (
            "internal_supply",
            "size_challenge",
            "grid_check",
            "finite_challenge",
        ):
            python(f"transport/{name}.py", "--out", "transport/results")
        library = "transport/lib_evaporation.so"
        command(
            env.get("NVCC", "nvcc"),
            "-O3",
            "-arch=" + env.get("CUDA_ARCH", "native"),
            "--shared",
            "-Xcompiler=-fPIC",
            "transport/four_precursors.cu",
            "-o",
            library,
        )
        python(
            "transport/four_precursors.py",
            "--library",
            library,
            "--out",
            "transport/results",
        )
        python(
            "transport/farnesene_pool.py",
            "--out",
            "transport/results/farnesene_pool_day.csv",
        )
        python(
            "transport/finite_challenge.py",
            "--diameter",
            "60",
            "--out",
            "transport/results/finite60",
        )
        python(
            "uncertainty/rescreen.py",
            "--library",
            library,
            "--out",
            "uncertainty/results",
        )
        python("uncertainty/finite.py", "--out", "uncertainty/results")
        python(
            "figure2_table.py",
            "uncertainty/results/model_ranges.csv",
            "figure2.csv",
        )
        python("replot.py", "--out", ".")

    if stage in ("all", "viscosity"):
        for name in ("predict", "curves", "constant_bound"):
            python(f"viscosity/{name}.py", "--out", "viscosity/results")
        for name in ("fractional", "marker_equivalent"):
            python(
                f"viscosity/fractional/{name}.py",
                "--packet",
                folder,
                "--out",
                "viscosity/fractional/results",
            )
        python(
            "viscosity/fractional/plot_fractional.py",
            "--out",
            "viscosity/fractional/results",
        )
        command(
            "bash",
            "viscosity/radial/replay.sh",
            folder,
            folder / "viscosity/radial",
        )
        python(
            "viscosity/ocimene_size.py", "--out", "viscosity/ocimene_size.csv"
        )

    if stage in ("all", "cloud"):
        command("bash", "parcel/replay.sh", folder / "parcel")
        python("density/cloud_density.py", "--out", "density/results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument(
        "folder",
        type=Path,
        help="External directory for staged sources and outputs",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=CHECKOUT / "docs/data/liu-terpene/digitized.zip",
    )
    args = parser.parse_args()
    reproduce(args.stage, args.folder.resolve(), args.archive.resolve())
