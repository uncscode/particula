"""Export published sample residuals and water activities as tidy CSV files."""

import csv
import json
from pathlib import Path


def export(root):
    """Export the saved results without repeating any AIOMFAC requests."""
    result = json.loads(
        (root / "website/results/mixture-activity/data.json").read_text()
    )
    out = root / "website/public/study-data/mt-hua"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sample-growth.csv").open("w", newline="") as stream:
        fields = (
            "key",
            "site",
            "date",
            "season",
            "observed",
            "zsr",
            "zsr_web_pure",
            "growth",
            "zsr_residual",
            "activity_residual",
            "mixing_effect",
            "sparse_grid",
        )
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(row for row in result["rows"] if "observed" in row)
    with (out / "water-activity.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "mixture_id",
                "temperature_K",
                "water_to_dry_mass",
                "water_activity",
            ]
        )
        for row in result["rows"]:
            for ratio, activity in zip(
                result["water_dry"], row["aw"], strict=True
            ):
                writer.writerow([row["key"], 298.15, ratio, activity])


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[2])
