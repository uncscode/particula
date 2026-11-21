"""Export the reported effects and their paired absolute endpoints."""

import csv
import json
from pathlib import Path


def export(root):
    """Publish one row per comparison, observable, control and time."""
    result = json.loads(
        (root / "website/results/regimes/data.json").read_text()
    )
    target = root / "website/public/study-data/mam4xx"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "paired-effects.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "regime",
                "process",
                "control",
                "observable",
                "hour",
                "absolute_difference",
                "percent_change",
            ]
        )
        for row in result["endpoints"]:
            for h, delta, pct in zip(
                row["hours"], row["difference"], row["percent"], strict=True
            ):
                writer.writerow(
                    [
                        row["regime"],
                        row["process"],
                        row["control"],
                        row["quantity"],
                        h,
                        delta,
                        pct,
                    ]
                )
    models = {
        "coarse": "brown",
        "charge": "diagnostic",
        "ion history": "dynamic",
        "dispersion": "dispersion",
        "dielectric + dispersion": "combined",
        "settling": "settling",
        "shear": "shear",
        "mode width": "modal",
    }
    costs = {
        (r["regime"], r["mechanism"]): r["elapsed"]
        for r in result["cases"] + result["modal_cases"]
        if r["control"] == "central"
    }
    absolute = {
        (r["regime"], r["process"]): r["difference"][-1]
        for r in result["endpoints"]
        if r["control"] == "central" and r["quantity"] == "uf_cm3"
    }
    with (target / "priorities.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "regime",
                "process",
                "change_cm3",
                "retention_change_percent",
                "range_min_percent",
                "range_max_percent",
                "grid_shift_points",
                "central_box_seconds",
            ]
        )
        for r in sorted(
            result["contrasts"],
            key=lambda row: (row["regime"], -abs(row["effect"])),
        ):
            reg, process = r["regime"], r["process"]
            writer.writerow(
                [
                    reg,
                    process,
                    absolute[reg, process],
                    r["effect"],
                    r["low"],
                    r["high"],
                    r["grid_shift"],
                    costs[reg, models[process]],
                ]
            )


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[2])
