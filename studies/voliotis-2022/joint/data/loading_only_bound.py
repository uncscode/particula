"""Calculate the minimum worst relative peak error for a loading-only model."""

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOSES = ("alpha_pinene_ppb", "o_cresol_ppb", "isoprene_ppb")


def estimate():
    """Summarize equal-dose groups with their missing and zero runs."""
    groups = defaultdict(list)
    with (HERE / "campaign_peaks.csv").open() as stream:
        for row in csv.DictReader(stream):
            groups[tuple(row[key] for key in DOSES)].append(row)
    summary = []
    for doses, records in groups.items():
        positive = [
            float(row["soa_peak_ug_m3"])
            for row in records
            if row["status"] == "positive"
        ]
        statuses = ["positive", "reported_zero", "missing"]
        runs = [
            " ".join(row["run"] for row in records if row["status"] == status)
            for status in statuses
        ]
        if positive:
            low, high = min(positive), max(positive)
            optimum = 2 * low * high / (low + high)
            floor = (high - low) / (high + low)
            values = (low, high, high / low, optimum, floor)
        else:
            values = ("",) * 5
        summary.append((*doses, *runs, *values))
    return summary


if __name__ == "__main__":
    with (HERE / "campaign_peak_bound.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                *DOSES,
                "positive_runs",
                "reported_zero_runs",
                "missing_runs",
                "min_positive_peak",
                "max_positive_peak",
                "peak_ratio",
                "minimax_common_peak",
                "minimax_max_relative_error",
            )
        )
        writer.writerows(estimate())
