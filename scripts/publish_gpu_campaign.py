"""Validate a complete CUDA campaign; export only displayed plotting values."""

import argparse
import json
from pathlib import Path

from scripts.gpu_studies.campaign import all_jobs, valid_result
from scripts.gpu_studies.worker import executable_hash


def plot_row(report):
    """Select only the small numeric series used by published line/bar plots.

    Use an allowlist so future simulation arrays or validation diagnostics
    cannot accidentally become downloadable data. Heatmaps and particle states
    belong to transient execution, not this plotting payload.
    """
    case, mode = report["case"], report["mode"]
    name = case["name"]
    scaling = name.startswith(("particles-", "boxes-", "brownian-"))
    occupancy = name.startswith("species-") and mode == "captured"
    if not (scaling or occupancy) or mode == "evolution":
        return None
    case_fields = ["name", "steps"]
    case_fields.append("boxes" if name.startswith("boxes-") else "slots")
    if occupancy:
        case_fields = ["name", "steps", "occupancy"]
    timing_fields = ("median_s", "minimum_s", "maximum_s")
    if occupancy:
        timing_fields = ("median_s",)
    result = {
        "case": {key: case[key] for key in case_fields},
        "mode": mode,
        "timings": {
            "resident": {
                key: report["timings"]["resident"][key] for key in timing_fields
            }
        },
    }
    if name.startswith("boxes-"):
        result["science"] = {
            "initial_active_slots": report["science"]["initial_active_slots"]
        }
    million = name == "particles-condensation-1048576"
    if million:
        result["timings"].update(
            {
                key: {"median_s": report["timings"][key]["median_s"]}
                for key in ("setup", "capture", "download", "end_to_end")
                if key in report["timings"]
            }
        )
    if (million and mode != "numpy") or (
        name.startswith("brownian-") and mode != "direct"
    ):
        result["memory"] = {
            "mempool_peak_bytes": report["memory"]["mempool_peak_bytes"]
        }
    if million and mode == "captured":
        for key in (
            "radius_edges_m",
            "initial_mean_histogram_number_m3",
            "final_mean_histogram_number_m3",
        ):
            result[key] = report[key]
    return result


def validate_samples(report, key):
    """Reject partial timing repetitions before publishing a ratio."""
    if report["warmups"] != 3 or report["samples"] != 7:
        raise ValueError(f"Incomplete repetitions: {key}")
    for timing in report["timings"].values():
        if len(timing["samples_s"]) != 7:
            raise ValueError(f"Missing timing samples: {key}")


def parity_evidence(directory, reports, source_hash):
    """Require full-array CPU equivalence for every published CPU comparison."""
    expected = {
        row["case"]["name"]: row["case"]
        for row in reports
        if row["mode"] == "numpy"
    }
    if not expected:
        return None
    parity = json.loads((directory / "cpu-parity.json").read_text())
    actual = {row["case"]["name"]: row["case"] for row in parity["cases"]}
    if (
        parity.get("qualified") is not True
        or parity.get("executable_sha256") != source_hash
        or actual != expected
    ):
        raise ValueError("Incomplete or stale CPU/CUDA parity evidence")
    return parity


def export_campaign(directory, destination):
    """Validate all jobs before selecting displayed point/line data."""
    manifest = json.loads((directory / "manifest.json").read_text())
    source_hash = executable_hash()
    if not manifest.get("qualified"):
        raise ValueError("Campaign is incomplete")
    reports, groups = [], {}
    for case, mode in all_jobs():
        key = f"{case.name}-{mode}"
        path = directory / f"{key}.json"
        if manifest["jobs"].get(key, {}).get(
            "status"
        ) != "passed" or not valid_result(path, case, mode, source_hash):
            raise ValueError(f"Unqualified or stale required result: {key}")
        report = json.loads(path.read_text())
        if mode != "evolution":
            validate_samples(report, key)
            groups.setdefault(case.name, []).append(report)
        reports.append(report)
    for name, rows in groups.items():
        if len({r["input_sha256"] for r in rows}) != 1:
            raise ValueError(f"Modes used different initial states: {name}")
        gpu = [r for r in rows if r["mode"] != "numpy"]
        if len({r["result_hash"] for r in gpu}) != 1:
            raise ValueError(f"CUDA modes changed continuation: {name}")
    parity = parity_evidence(directory, reports, source_hash)
    gpu = next(row for row in reports if row["mode"] == "captured")
    output = {
        "schema_version": 2,
        "qualified": True,
        "campaign_started_at": manifest["started_at"],
        "executable_sha256": source_hash,
        "job_count": len(reports),
        "cross_mode_cuda_exact": True,
        "cpu_cuda_parity_case_count": len(parity["cases"]) if parity else 0,
        "provenance": {
            key: gpu[key]
            for key in ("gpu_name", "warp", "cpu_model", "source_revision")
        },
        "memory_method": gpu["memory"]["method"],
        "reports": [row for report in reports if (row := plot_row(report))],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


def main():
    """Export only after all original campaign jobs are qualified."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report = export_campaign(args.directory, args.destination)
    print(
        f"Validated {report['job_count']} jobs; "
        f"exported {len(report['reports'])} plotted series"
    )


if __name__ == "__main__":
    main()
