"""Serial, process-isolated scaling campaign with explicit time/VRAM budgets.

Run ``python -m scripts.gpu_studies.campaign --output .artifacts/gpu-campaign``.
Successful matching cases can resume; failures/timeouts remain visible and do
not become smaller substitute workloads. Large raw outputs stay outside Git.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.gpu_studies.cases import StudyCase
from scripts.gpu_studies.worker import ROOT, executable_hash


def timing_cases():
    """Declare accepted sizes, independent species widths and occupancy."""
    cases = []
    for family in ("condensation", "dilution"):
        for slots in (1024, 16384, 131072, 1048576):
            cases.append(
                StudyCase(f"particles-{family}-{slots}", family, 1, slots)
            )
    for boxes in (1, 8, 64, 512, 1024):
        cases.append(StudyCase(f"boxes-{boxes}", "condensation", boxes, 1024))
    for slots in (128, 512, 2048, 8192, 16384):
        cases.append(
            StudyCase(
                f"brownian-{slots}",
                "brownian",
                1,
                slots,
                time_step=1.0,
                steps=1,
            )
        )
    for occupancy in (0.25, 0.75, 1.0):
        for family in ("condensation", "brownian"):
            cases.append(
                StudyCase(
                    f"species-{family}-{int(occupancy * 100)}",
                    family,
                    8,
                    2048,
                    materials=4,
                    gases=5,
                    occupancy=occupancy,
                    time_step=0.5,
                    steps=2,
                )
            )
    cases.append(
        StudyCase(
            "heterogeneous-64",
            "condensation",
            64,
            4096,
            occupancy=0.75,
            heterogeneous=True,
        )
    )
    # 1,572,864 active particles, with two canonical free slots for each source.
    cases.append(
        StudyCase(
            "full-million",
            "full",
            4096,
            512,
            occupancy=0.75,
            time_step=1.0,
            steps=2,
        )
    )
    return tuple(cases)


def timing_jobs():
    """Separate modes so each measured run has an uncontended GPU process."""
    jobs = []
    for case in timing_cases():
        modes = ["direct", "prepared", "captured"]
        if case.family in ("condensation", "dilution"):
            modes.insert(0, "numpy")
        if case.name.startswith("brownian-"):
            modes.append("legacy")
        jobs.extend((case, mode) for mode in modes)
    return tuple(jobs)


def all_jobs():
    """Include all required scientific histories and the large lifecycle."""
    studies = (
        StudyCase(
            "heterogeneous-evolution",
            "condensation",
            64,
            4096,
            occupancy=0.75,
            time_step=1e-6,
            steps=6000,
            heterogeneous=True,
            experiment="heterogeneous",
        ),
        StudyCase(
            "time-varying-16",
            "condensation",
            16,
            4096,
            materials=4,
            gases=5,
            occupancy=0.75,
            time_step=1e-6,
            steps=6000,
            heterogeneous=True,
            experiment="forcing",
        ),
        StudyCase(
            "stochastic-16-seeds",
            "brownian",
            16,
            1024,
            time_step=1.0,
            steps=40,
            heterogeneous=True,
            experiment="stochastic",
        ),
        StudyCase(
            "million-lifecycle",
            "full",
            4096,
            512,
            occupancy=0.75,
            time_step=1.0,
            steps=4,
            experiment="lifecycle",
        ),
    )
    return timing_jobs() + tuple((case, "evolution") for case in studies)


def save(path, report):
    """Publish a complete manifest atomically after each attempted job."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def valid_result(path, case, mode, source_hash):
    """Reject absent, corrupt or mismatched worker output on run and resume."""
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(result, dict) and (
        result.get("qualified") is True
        and result.get("executable_sha256") == source_hash
        and result.get("case") == asdict(case)
        and result.get("mode") == mode
    )


def campaign(  # noqa: C901 -- serial admission, recovery and budget state
    output, *, hours=8.0, case_seconds=600, legacy_seconds=600, select=None
):
    """Measure complete declared jobs and retain unsuccessful entries."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    now = datetime.now(timezone.utc)
    if manifest.exists():
        report = json.loads(manifest.read_text())
    else:
        report = {
            "schema_version": 1,
            "started_at": now.isoformat(),
            "hours": hours,
            "jobs": {},
            "qualified": False,
        }
    deadline = (
        datetime.fromisoformat(report["started_at"]).timestamp()
        + report["hours"] * 3600
    )
    source_hash = executable_hash()
    for case, mode in all_jobs():
        key = f"{case.name}-{mode}"
        destination = output / f"{key}.json"
        previous = report["jobs"].get(key, {})
        if previous.get("status") == "passed" and valid_result(
            destination, case, mode, source_hash
        ):
            continue
        report["jobs"][key] = {
            "case": asdict(case),
            "mode": mode,
            "status": "pending",
        }
    save(manifest, report)
    for case, mode in all_jobs():
        key = f"{case.name}-{mode}"
        row = report["jobs"][key]
        if row["status"] == "passed" or (
            select is not None and select not in key
        ):
            continue
        remaining = deadline - time.time()
        if remaining <= 0:
            row["status"] = "budget_exhausted"
            save(manifest, report)
            break
        limit = min(
            remaining, legacy_seconds if mode == "legacy" else case_seconds
        )
        destination = output / f"{key}.json"
        destination.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.gpu_studies.worker",
            "--case",
            json.dumps(asdict(case)),
            "--mode",
            mode,
            "--output",
            str(destination.resolve()),
        ]
        print(f"RUN {key} (limit {limit:.0f} s)", flush=True)
        row.update(status="running", command=command, limit_seconds=limit)
        save(manifest, report)
        started = time.perf_counter()
        with (output / f"{key}.log").open("w") as log:
            try:
                completed = subprocess.run(  # noqa: S603 -- fixed local worker
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=limit,
                    check=False,
                )
                row.update(
                    status="passed" if completed.returncode == 0 else "failed",
                    returncode=completed.returncode,
                )
                if row["status"] == "passed" and not valid_result(
                    destination, case, mode, source_hash
                ):
                    row.update(
                        status="failed",
                        reason="missing or mismatched result provenance",
                    )
            except subprocess.TimeoutExpired:
                row["status"] = "timed_out"
        row["elapsed_seconds"] = time.perf_counter() - started
        save(manifest, report)
        print(
            f"{row['status'].upper()} {key}: {row['elapsed_seconds']:.1f} s",
            flush=True,
        )
    report["qualified"] = all(
        row["status"] == "passed" for row in report["jobs"].values()
    )
    report["executable_sha256"] = source_hash
    save(manifest, report)
    return report


def main():
    """Run/resume the fixed matrix; selected subsets never qualify the whole."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--case-seconds", type=float, default=600.0)
    parser.add_argument("--legacy-seconds", type=float, default=600.0)
    parser.add_argument("--select")
    args = parser.parse_args()
    if min(args.hours, args.case_seconds, args.legacy_seconds) <= 0:
        parser.error("budgets must be positive")
    result = campaign(
        args.output,
        hours=args.hours,
        case_seconds=args.case_seconds,
        legacy_seconds=args.legacy_seconds,
        select=args.select,
    )
    if not result["qualified"]:
        print("Campaign incomplete: required cases remain unfinished.")


if __name__ == "__main__":
    main()
