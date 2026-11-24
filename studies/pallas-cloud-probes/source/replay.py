"""Reproduce the Pallas probe response and manuscript tables from Zenodo v3."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

SOURCE = Path(__file__).resolve().parent
ARCHIVE = "https://zenodo.org/records/19220993/files"
# The authors supply MD5 checksums for each versioned file.
CHECKSUMS = {
    "FMI.CAPS.2022.metadata.csv": "f56057a931e678163b6703bdc20d5c5f",
    "FMI.CAS.b1.20220913.csv": "fc3143139a6e9facb95487f6aeca86f4",
    "FMI.CDA.2022.metadata.csv": "d30e8b9b3ce386eda76f3a1188e4a549",
    "FMI.CDA.b1.20220914.csv": "463d17906aea80fee4406d4663b57b7e",
    "FMI.FSSP.2022.metadata.csv": "bfbae705a1fecf998d6c2e6972af3933",
    "FMI.FSSP.b1.20220913.csv": "562cc1731e83646e660e147a80443b45",
    "FMI.FSSP.b1.20220913.nc": "44da7eede3b6de634b1f4f1df941d40c",
    "OULU.ICEMET.2022.metadata.csv": "37fcd109fd3f9b3bba8ad07f23855af2",
    "OULU.ICEMET.b1.20220914.csv": "ecc008f5c469023f5a7d11c283832121",
    "OULU.ICEMET.b1.20220914.nc": "6f602177b203219ad9c9342f1eaade2f",
}


def check_input(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def fetch(directory, local):
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKSUMS.items():
        dest = directory / name
        if dest.exists():
            if check_input(dest) != expected:
                raise ValueError(
                    f"Archive checksum mismatch for existing {dest}"
                )
            continue
        origin = (
            (local / name).open("rb")
            if local
            else urlopen(f"{ARCHIVE}/{name}", timeout=120)
        )
        # Partial transfers cannot become published inputs.
        with origin, tempfile.TemporaryDirectory(dir=directory) as pending:
            temp = Path(pending) / name
            with temp.open("wb") as stream:
                shutil.copyfileobj(origin, stream)
            if check_input(temp) != expected:
                raise ValueError(f"Archive checksum mismatch for {name}")
            temp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work", type=Path, required=True, help="Replay directory"
    )
    parser.add_argument(
        "--input-dir", type=Path, help="Use existing Zenodo files"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "fetch", "field", "optics", "spectra"],
        default="all",
    )
    args = parser.parse_args()
    run_dir = args.work.resolve()
    if run_dir.is_relative_to(SOURCE):
        parser.error("Choose a work directory outside this source directory.")
    fetch(run_dir / "input", args.input_dir)
    if args.stage == "fetch":
        return

    optics = run_dir / "optics"
    spectra = run_dir / "spectra"
    optics.mkdir(exist_ok=True)
    spectra.mkdir(exist_ok=True)
    checkout = SOURCE.parents[2]
    paths = [str(SOURCE / "optics"), str(SOURCE)]
    if (checkout / "particula/__init__.py").exists():
        paths.append(str(checkout))
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(paths),
        PALLAS_RUN_DIR=str(run_dir),
        OPENBLAS_NUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONDONTWRITEBYTECODE="1",
    )
    env.setdefault("MPLCONFIGDIR", str(run_dir / ".mpl-cache"))
    env.setdefault("WARP_CACHE_PATH", str(run_dir / ".warp-cache"))

    def run(script, *arguments):
        print(f"Running {script} {' '.join(map(str, arguments))}", flush=True)
        subprocess.run(
            [sys.executable, SOURCE / script, *arguments],
            cwd=run_dir,
            env=env,
            check=True,
        )

    if args.stage in ("all", "field"):
        run("study.py", "pilot")
        run("calibration.py")
        run("study.py", "analyze")
        for scale in ("pilot", "full"):
            run("response.py", scale)
        for script in (
            "challenge",
            "transfer",
            "bands",
            "independent",
            "matched",
            "support",
            "plot",
            "verify",
        ):
            run(f"{script}.py")
    if args.stage in ("all", "optics"):
        for scale in ("pilot", "main", "refined", "convergence"):
            run("optics/simulate.py", scale, "--out", optics)
        for script in ("score", "plot", "verify"):
            run(f"optics/{script}.py", optics)
    if args.stage in ("all", "spectra"):
        for script in ("transfer", "budget", "method_only", "figure"):
            run(f"spectra/{script}.py", spectra)
        run("check_manuscript.py", spectra)


if __name__ == "__main__":
    main()
