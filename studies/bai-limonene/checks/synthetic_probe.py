"""Sample July parcel columns through observed probe diameter bins."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from probes import CHECKSUMS, median_diameter, read_probe


def sampled(states, bounds):
    """Calculate two sensor medians for each time and sampling height."""
    rows = []
    for path in states:
        with np.load(path) as data:
            heights = data["height_m"]
            dz = np.diff(heights)
            if heights[0] != 0 or heights[-1] != 100 or (dz <= 0).any():
                raise ValueError(f"Invalid sampling heights in {path}")
            weights = np.r_[0, dz] + np.r_[dz, 0]
            weights /= weights.sum()
            for j, hour in enumerate(data["hour_in_week"]):
                diameter = data["wet_radius_m"][j].astype(float) * 2e6
                number = data["number_m3"][j].astype(float) / 1e6
                for instrument, edges in bounds.items():
                    by_height = np.array(
                        [
                            np.histogram(sizes, edges, weights=counts)[0]
                            for sizes, counts in zip(
                                diameter, number, strict=True
                            )
                        ]
                    )
                    for window, counts in [
                        ("cloud_top", by_height[-1]),
                        ("whole_cloud", weights @ by_height),
                    ]:
                        rows.append(
                            {
                                "hour": hour,
                                "instrument": instrument,
                                "window": window,
                                "median_um": median_diameter(counts, edges),
                            }
                        )
    return pd.DataFrame(rows)


def run(run_dir, cache, output, smoke):
    """Compare paired sun-aged and dark columns on a common solar clock."""
    cache.mkdir(parents=True, exist_ok=True)
    bounds = {}
    for name in CHECKSUMS:
        edges, _ = read_probe(cache, name)
        bounds[name] = edges[(edges >= 3.5) & (edges <= 40)]
    label = "smoke" if smoke else "resolved"
    columns = run_dir / "scratch/columns"
    paired = []
    for month, mode in [(0, "dark"), (7, "ranked")]:
        states = sorted(
            columns.glob(f"{label}_m{month:02d}_clean_w0.25_{mode}_*.npz")
        )
        if not states:
            raise FileNotFoundError(
                f"Missing {label} columns in {columns}. Run the cloud replay."
            )
        paired.append(sampled(states, bounds))
    dark, light = paired
    keys = ["instrument", "window"]
    merged = light.merge(
        dark.drop(columns="hour"),
        on=keys,
        suffixes=("", "_dark"),
        validate="many_to_one",
    )
    records = []
    for (instrument, window), group in merged.groupby(keys):
        group = group.sort_values("hour")
        hours = group.hour.to_numpy()
        expected = [12] if smoke else [*range(0, 168, 3), 167]
        np.testing.assert_array_equal(hours, expected)
        mid = (hours[1:] + hours[:-1]) / 2
        time_weight = np.diff(np.r_[-0.5, mid, 167.5])
        delta = np.average(
            group.median_um - group.median_um_dark, weights=time_weight
        )
        records.append(
            dict(
                instrument=instrument,
                window=window,
                samples=len(hours),
                delta_median_um=delta,
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(output / f"synthetic-probes-{label}.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the single matched July week-hour 12 from the smoke replay",
    )
    args = parser.parse_args()
    run(args.run_dir, args.cache, args.output, args.smoke)
