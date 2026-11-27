"""Compare Brownian ensemble means by timestep and particle count."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def read(root, files=None):
    """Keep family and realization IDs distinct."""
    cases = {}
    for file in sorted(root.glob("*.csv") if files is None else files):
        found = re.fullmatch(
            r"(coagulation|coupled)-k(\d+)-s(\d+)-seed(\d+)", file.stem
        )
        if found is None:
            continue
        mode, slots, steps, seed = found.groups()
        group = cases.setdefault((mode, int(slots), int(steps)), {})
        with file.open() as stream:
            for row in csv.DictReader(stream):
                key = (
                    int(row["family"]),
                    int(row["density_multiplier"]),
                    int(row["mix"]),
                    row["leg"],
                )
                group.setdefault(key, {})[int(seed)] = np.array(
                    [
                        float(row[k])
                        for k in (
                            "number_fraction",
                            "mean_wet_diameter_um",
                            "water_dry_ratio",
                        )
                    ]
                )
    return cases


def compare(root, out):
    """Report refinement changes and seed errors per family."""
    cases = read(root)
    records = []
    for (mode, k, s), group in sorted(cases.items()):
        for parameter, other in (
            ("steps", (mode, k, s * 2)),
            ("slots", (mode, k * 2, s)),
        ):
            if other not in cases:
                continue
            for key, current in sorted(group.items()):
                refined = cases[other].get(key, {})
                seeds = sorted(current.keys() & refined.keys())
                if len(seeds) < 2:
                    continue
                a = np.stack([current[n] for n in seeds])
                b = np.stack([refined[n] for n in seeds])
                diff = a - b
                se = diff.std(axis=0, ddof=1) / np.sqrt(len(seeds))
                row = dict(
                    mode=mode,
                    slots=k,
                    steps=s,
                    refined_parameter=parameter,
                    family=key[0],
                    density=key[1],
                    mixture=key[2],
                    leg=key[3],
                    seeds=len(seeds),
                )
                for j, unit in enumerate(
                    (
                        "number_fraction",
                        "mean_wet_diameter_um",
                        "water_dry_ratio",
                    )
                ):
                    row[unit + "_refined_mean"] = b[:, j].mean()
                    row[unit + "_change"] = diff[:, j].mean()
                    row[unit + "_change_se"] = se[j]
                records.append(row)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=records[0], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    print("Comparisons", len(compare(**vars(p.parse_args()))))
