"""Verify the archived SABRE inputs and repeat the optical and chemical fits."""

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from figure_archive import restore_digitizations

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "raw"
LAB_PDF = "Gaston-2014.pdf"
LAB_HASH = "d7303b4575b59271e117bb62e06d947cfd2809e3ccf51207e9c426b28c0150ab"


def records(name):
    """Read a source manifest, including its archive filenames and hashes."""
    with (HERE / name).open(newline="") as source:
        return list(csv.DictReader(source))


def selected_files():
    """Select final optical flights and associated chemical observations."""
    inventory = records("raw-manifest.csv")
    dates = {
        row["flight_utc"]
        for row in inventory
        if row["data_id"] == "SABRE-SOAP"
        and row["revision"] == "R0"
        and row["flight_utc"].startswith("2023")
    }
    optical = [
        row
        for row in inventory
        if row["flight_utc"] in dates and row["revision"] in {"R3", "R0"}
    ]
    other = records("chemistry/chemistry-sources.csv") + records(
        "chemistry/composition-sources.csv"
    )
    return optical, other


def verify_files(rows):
    """Report all missing or changed files before starting the calculation."""
    errors = []
    for row in rows:
        path = RAW / row["filename"]
        if not path.is_file():
            errors.append(f"missing {path.name}")
            continue
        if "bytes" in row and path.stat().st_size != int(row["bytes"]):
            errors.append(f"size mismatch {path.name}")
            continue
        with path.open("rb") as source:
            if (
                hashlib.file_digest(source, "sha256").hexdigest()
                != row["sha256"]
            ):
                errors.append(f"SHA-256 mismatch {path.name}")
    if errors:
        raise SystemExit(
            "Check the originals in " + str(RAW) + "\n" + "\n".join(errors)
        )


def run(script, *args):
    """Run each calculation with this checkout and the selected interpreter."""
    env = os.environ.copy()
    roots = [str(ROOT), str(ROOT / "packages/particula-aiomfac/src")]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    print(f"Running {script}", flush=True)
    subprocess.run(  # noqa: S603
        [sys.executable, str(HERE / script), *args],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main():
    """Build selection, fits, chemistry controls and matched-group contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="list required files"
    )
    parser.add_argument(
        "--check-inputs", action="store_true", help="verify inputs and stop"
    )
    options = parser.parse_args()
    optical, other = selected_files()
    required = [*optical, *other, {"filename": LAB_PDF, "sha256": LAB_HASH}]
    if options.list:
        print("\n".join(row["filename"] for row in required))
        return
    verify_files(required)
    print(f"Verified {len(required) - 1} ICARTT files and {LAB_PDF}.")
    if options.check_inputs:
        return

    restore_digitizations()

    # The join discovers flights from filenames. Stage exactly the pinned files
    # so unrelated archive products beside a user's originals cannot enter it.
    with TemporaryDirectory(dir=RAW) as staging:
        subset = Path(staging)
        for row in optical:
            shutil.copyfile(RAW / row["filename"], subset / row["filename"])
        run("join_flights.py", "--raw", str(subset))
        (subset / "joined.npz").replace(RAW / "joined.npz")
    run("chemistry/evidence/extract_gaston.py", str(RAW / LAB_PDF))
    for script in (
        "analysis.py",
        "flight_analysis.py",
        "chemistry/join_chemistry.py",
        "chemistry/join_composition.py",
        "chemistry/hydration.py",
        "chemistry/cold_acid_check.py",
        "chemistry/composition_grid.py",
        "chemistry/field_fit.py",
        "chemistry/joint_fit.py",
        "chemistry/native_channels.py",
        "chemistry/evidence/fit_laboratory.py",
        "integrate.py",
        "compile_figure_values.py",
    ):
        run(script)
    print(f"Group contrasts and fitted shares are in {HERE}.")


if __name__ == "__main__":
    main()
