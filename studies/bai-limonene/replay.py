"""Build a disposable Bai study run from versioned inputs and model source."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FMI_FINGERPRINT = (
    "3ab4a618a2d5eb5ebf5cbc274a4bddeb7aec3231cc4f66066ff972344cfef121"
)
CASES = [
    "--label",
    "resolved",
    "--columns",
    "--midpoint",
    "--merge-physical",
    "--bins",
    "768",
    "--subbins",
    "32",
    "--sub",
    "60",
    "--probes",
    "36",
    "--dt",
    ".025",
    "--levels",
    "100",
]


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(path, spec):
    if sha256(path) == spec["sha256"]:
        return
    if "data_sha256" in spec:
        # NASA can reserialize the JSON envelope. Verify the exact hourly values.
        data = json.loads(path.read_text())["properties"]["parameter"]
        fixed = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(fixed).hexdigest() == spec["data_sha256"]:
            return
    raise ValueError(f"Source fingerprint mismatch for {path}")


def fetch(folder, specs=None):
    if specs is None:
        specs = json.loads((HERE / "sources.json").read_text())
    for name, spec in specs.items():
        out = folder / name
        if out.exists():
            verify(out, spec)
            continue
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            candidate = Path(temp) / name
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "3",
                    "--max-time",
                    "180",
                    "--output",
                    str(candidate),
                    spec["url"],
                ],
                check=True,
            )
            verify(candidate, spec)
            candidate.replace(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "work",
        type=Path,
        help="Disposable cache and working directory outside the checkout",
    )
    parser.add_argument(
        "stages",
        nargs="+",
        choices=[
            "inputs",
            "smoke",
            "july-week",
            "solar-controls",
            "cloud",
            "radiation",
            "film-parcels",
            "cuda-parity",
        ],
    )
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    work = args.work.resolve()
    if work.is_relative_to(HERE.parents[1]) or HERE.is_relative_to(work):
        parser.error(
            "Choose a dedicated work directory outside the source directory."
        )
    if args.threads < 1:
        parser.error("Use at least one worker.")
    # Code is staged next to disposable native products so original relative
    # source-to-result relationships are preserved without polluting the checkout.
    work.mkdir(parents=True, exist_ok=True)
    marker = work / ".bai-limonene-replay"
    if not marker.is_file():
        if any(work.iterdir()):
            parser.error(
                "Choose an empty run directory or one initialized by this replay."
            )
        marker.write_text("Particula Bai literature study replay\n")
    shutil.copytree(
        HERE / "model",
        work,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    downloads = work / "downloads"
    scratch = work / "scratch"
    for path in [
        downloads,
        scratch / "columns",
        scratch / "radiation",
        work / "results",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(HERE.parents[1]), os.environ.get("PYTHONPATH", "")]
        ),
        SEASONAL_TMP=str(scratch),
        BAI_PARCEL_LIB=str(scratch / "parcel_cpu.so"),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS=str(args.threads),
        MPLCONFIGDIR=str(scratch / "mpl-cache"),
    )

    def run(*cmd):
        subprocess.run(list(map(str, cmd)), check=True, cwd=work, env=env)

    def python(rel, *args):
        run(sys.executable, work / rel, *args)

    def native(src, lib):
        run(
            os.getenv("CXX", "g++"),
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-fopenmp",
            work / src,
            "-o",
            scratch / lib,
        )

    def parcel(*args):
        python("seasonal/experiment.py", *args)

    for stage in args.stages:
        if stage == "inputs":
            fetch(downloads)
            python(
                "film/extract.py",
                downloads / "bai.pdf",
                downloads / "bai-si.pdf",
            )
            python(
                "sunlight/solar/solar.py",
                downloads / "bai.pdf",
                downloads / "bai-si.pdf",
            )
            python("sunlight/field-uv/extract.py", downloads / "bohn.pdf")
            python("sunlight/inflow/build_inflows.py", downloads)
            python("seasonal/forcing/fmi_fetch.py", downloads / "fmi")
            python(
                "seasonal/forcing/build.py",
                "--fmi-cache",
                downloads / "fmi",
                "--power-json",
                downloads / "power.json",
            )
            if (
                sha256(work / "seasonal/forcing/fmi_manifest.csv")
                != FMI_FINGERPRINT
            ):
                raise ValueError(
                    "The FMI records differ from the verified 120-day study data."
                )
            for name in ["astmg173.csv", "water_hale.yml"]:
                shutil.copy2(
                    downloads / name, work / "seasonal/radiation" / name
                )
            python("sunlight/optics/mie_dose.py", "--device", "cuda:0")
        elif stage in ("smoke", "cloud", "cuda-parity"):
            native("seasonal/transport.cpp", "aging.so")
            native("seasonal/parcel_cpu.cpp", "parcel_cpu.so")
            if stage == "cloud":
                parcel(*CASES, "--months", "0", "--modes", "dark")
                parcel(
                    *CASES,
                    "--months",
                    "7",
                    "--modes",
                    "ranked",
                    "--stride",
                    "3",
                )
                parcel(
                    *CASES,
                    "--months",
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    8,
                    9,
                    10,
                    11,
                    12,
                    "--modes",
                    "ranked",
                    "--stride",
                    "6",
                )
                parcel(
                    *CASES,
                    "--months",
                    "7",
                    "--modes",
                    "secondary",
                    "mixed",
                    "inverse",
                    "--stride",
                    "6",
                )
                for filename in [
                    "resolved_summary",
                    "size_spectra",
                    "matched_cohorts",
                ]:
                    python("seasonal/" + filename + ".py")
            elif stage == "smoke":
                for month, mode in [(0, "dark"), (7, "ranked")]:
                    parcel(
                        *CASES,
                        "--label",
                        "smoke",
                        "--months",
                        month,
                        "--modes",
                        mode,
                        "--profiles",
                        "clean",
                        "--wind",
                        "0.25",
                        "--hours",
                        "12",
                    )
                python("check_smoke.py")
            else:
                run(
                    os.getenv("NVCC", "nvcc"),
                    "-arch=native",
                    "-O3",
                    "-shared",
                    "-Xcompiler=-fPIC",
                    work / "sunlight/cloud_base.cu",
                    "-o",
                    scratch / "parcel_cuda.so",
                )
                for label in ("cpu", "cuda"):
                    env["BAI_PARCEL_LIB"] = str(
                        scratch / ("parcel_" + label + ".so")
                    )
                    parcel(
                        "--label",
                        "parity-" + label,
                        "--months",
                        7,
                        "--modes",
                        "ranked",
                        "--profiles",
                        "clean",
                        "--hours",
                        "12",
                        "--midpoint",
                        "--bins",
                        72,
                        "--subbins",
                        3,
                        "--sub",
                        30,
                        "--probes",
                        100,
                        "--zoom",
                        48,
                        "--dt",
                        0.05,
                        "--levels",
                        50,
                    )
                env["BAI_PARCEL_LIB"] = str(scratch / "parcel_cpu.so")
                python("check_smoke.py", "--parity")
        elif stage == "july-week":
            native("seasonal/transport.cpp", "aging.so")
            native("seasonal/parcel_cpu.cpp", "parcel_cpu.so")
            for month, mode in [(0, "dark"), (7, "ranked")]:
                parcel(
                    *CASES,
                    "--label",
                    "resolved-week",
                    "--months",
                    month,
                    "--modes",
                    mode,
                    "--profiles",
                    "clean",
                    "--wind",
                    0.25,
                    "--stride",
                    3,
                )
            native("seasonal/radiation/transfer.cpp", "radiation/transfer.so")
            python("seasonal/radiation/sphere_table.py")
            python(
                "seasonal/radiation/seasonal_flux.py",
                "--prefix",
                "resolved-week",
                "--tag",
                "july-week",
                "--months",
                7,
                "--profiles",
                "clean",
                "--updrafts",
                0.25,
                "--models",
                "ranked",
                "--redo",
            )
            python("check_week.py")
        elif stage == "radiation":
            native("seasonal/radiation/transfer.cpp", "radiation/transfer.so")
            python("seasonal/radiation/sphere_table.py")
            for m, modes in [
                (list(range(1, 13)), ["ranked"]),
                ([7], ["secondary", "mixed", "inverse"]),
            ]:
                python(
                    "seasonal/radiation/seasonal_flux.py",
                    "--tag",
                    "resolved",
                    "--prefix",
                    "resolved",
                    "--months",
                    *m,
                    "--models",
                    *modes,
                    "--redo",
                )
            python("seasonal/radiation/refinement_summary.py")
        elif stage == "solar-controls":
            extra = json.loads(
                (HERE / "solar-control-sources.json").read_text()
            )
            fetch(downloads, extra)
            python(
                "seasonal/forcing/sensitivity.py",
                "--raw-ozone",
                downloads / "ozone.dat",
            )
            python(
                "seasonal/radiation/clear_inputs.py",
                downloads / "clear-sky.json",
            )
            native("seasonal/transport.cpp", "aging.so")
            native("seasonal/parcel_cpu.cpp", "parcel_cpu.so")
            native("seasonal/radiation/transfer.cpp", "radiation/transfer.so")
            python("seasonal/radiation/sphere_table.py")
            python("solar_controls.py")
        elif stage == "film-parcels":
            python(
                "campaign.py",
                "screen",
                work / "results/screen.csv",
                "--device",
                "cuda:0",
            )
            python("transfer.py", "structure", "--out", work / "results")


if __name__ == "__main__":
    main()
