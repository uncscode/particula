"""Calculate detection rates from the archived synthetic decisions."""

import pandas as pd
from scipy.stats import binomtest
from scripts.antarctic_ccn.model import DATA


def detection_rates():
    """Return pooled false attribution and effect-binned film recovery."""
    records = pd.read_csv(DATA / "synthetic-decisions.csv.gz")
    rows = []
    for noise, group in records.groupby("noise"):
        detected = group.qualified.str.startswith(
            ("finite", "film-unspecified")
        )
        effect = group.chemical.abs()
        categories = [
            ("false attribution pooled", 0, ~group.surface),
            ("correct", 3, group.surface & (effect >= 3) & (effect < 6)),
            ("correct", 6, group.surface & (effect >= 6) & (effect < 10)),
        ]
        for truth, low, mask in categories:
            successes = int(detected[mask].sum())
            count = int(mask.sum())
            limits = binomtest(successes, count).proportion_ci(method="wilson")
            rows.append(
                dict(
                    noise=noise,
                    truth=truth,
                    effect_lo=low,
                    k=successes,
                    n=count,
                    rate=successes / count,
                    wilson_lo=limits.low,
                    wilson_hi=limits.high,
                )
            )
    return rows
