"""Export published sample residuals and water activities as tidy CSV files."""

import csv
import json
from pathlib import Path

SALTS = (
    "(NH4)2SO4",
    "NH4NO3",
    "NH4Cl",
    "Na2SO4",
    "NaNO3",
    "NaCl",
    "K2SO4",
    "KNO3",
    "KCl",
    "CaSO4",
    "Ca(NO3)2",
    "CaCl2",
    "MgSO4",
    "Mg(NO3)2",
    "MgCl2",
)


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

    pure = {r["key"]: r for r in result["rows"] if r["key"].startswith("pure_")}
    with (out / "dry-composition.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "pure_control_id",
                "salt",
                "dry_volume_fraction",
                "dry_mass_fraction",
                "dry_salt_density_kg_m3",
                "pure_literature_growth",
                "pure_web_growth",
            ]
        )
        for row in result["rows"]:
            if "observed" not in row:
                continue
            for i, fraction in enumerate(row["dry_volume_fraction"]):
                control = pure[f"pure_{i}"]
                writer.writerow(
                    [
                        row["key"],
                        control["key"],
                        SALTS[i],
                        fraction,
                        fraction * control["density"] / row["density"],
                        control["density"],
                        control["zsr"],
                        control["growth"],
                    ]
                )


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[2])
