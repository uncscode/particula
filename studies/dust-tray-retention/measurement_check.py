"""Propagate specified diameter and weak-channel scenarios through measured D50."""

import csv
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open() as source:
        return list(csv.DictReader(source))


def median(diameter, shares):
    cdf = np.cumsum(shares) / shares.sum()
    return np.exp(np.interp(0.5, cdf, np.log(diameter)))


def perturbed_medians(diameter, shares, removed, hidden_pct, scale):
    """Either extreme placement bounds the median under a specified hidden share."""
    kept = np.where(shares < removed, 0.0, shares)
    kept *= (100.0 - hidden_pct) / kept.sum()
    medians = []
    for channel in (0, -1):
        candidate = kept.copy()
        candidate[channel] += hidden_pct
        medians.append(median(diameter * scale, candidate))
    return medians


def main():
    channels = read("channels.csv")
    totals = read("total-mass.csv")
    rows = []
    for mass in totals:
        wind = mass["wind_ms"]
        group = [row for row in channels if row["wind_ms"] == wind]
        diameter, early, late = (
            np.asarray([float(row[field]) for row in group])
            for field in ("channel_um", "early_pct", "measured_pct")
        )
        measured = median(diameter, late) - median(diameter, early)
        assert np.isclose(early.sum(), 100.0) and np.isclose(late.sum(), 100.0)
        # Independent Welch interval for total mass, conditional on the sampled trays.
        sem = np.asarray(
            [float(mass[k]) for k in ("early_sem_mg", "measured_sem_mg")]
        )
        count = np.asarray([int(mass[k]) for k in ("early_n", "late_n")])
        variance = sem**2
        df = variance.sum() ** 2 / (variance**2 / (count - 1)).sum()
        halfwidth = t.ppf(0.975, df) * np.sqrt(variance.sum())
        delta_mass = float(mass["measured_mg"]) - float(mass["early_mg"])
        for case, scale_error, removed, hidden_pct in (
            ("original", 0, 0, 0),
            ("calibrant-scale", 0.01, 0, 0),
            ("reference-drift-scale", 0.02, 0, 0),
            ("weak-channel-0.1", 0, 0.1, 0),
            ("weak-channel-0.5", 0, 0.5, 0),
            ("unreported-total-1pct", 0, 0, 1),
            ("combined-stress", 0.02, 0.5, 1),
        ):
            # Every endpoint may shift in either direction. The worst signed
            # change uses opposite errors at the two sample times.
            endpoints = []
            for spectrum in (early, late):
                medians = []
                for scale in (1 - scale_error, 1 + scale_error):
                    medians.extend(
                        perturbed_medians(
                            diameter, spectrum, removed, hidden_pct, scale
                        )
                    )
                endpoints.append(medians)
            differences = [b - a for a, b in product(*endpoints)]
            low, high = min(differences), max(differences)
            if case == "original":
                assert np.isclose(low, measured) and np.isclose(high, measured)
            rows.append(
                dict(
                    wind_ms=wind,
                    case=case,
                    diameter_scale_error_pct=scale_error * 100,
                    censor_below_channel_pp=removed,
                    assumed_hidden_total_pct=hidden_pct,
                    early_removed_pct=early[early < removed].sum(),
                    late_removed_pct=late[late < removed].sum(),
                    measured_delta_d50_um=measured,
                    delta_d50_min_um=low,
                    delta_d50_max_um=high,
                    early_trays=count[0],
                    late_trays=count[1],
                    delta_total_mass_mg=delta_mass,
                    delta_total_mass_se_mg=np.sqrt(variance.sum()),
                    delta_total_mass_welch_df=df,
                    delta_total_mass_95_low_mg=delta_mass - halfwidth,
                    delta_total_mass_95_high_mg=delta_mass + halfwidth,
                )
            )
    with (ROOT / "measurement-sensitivity.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
