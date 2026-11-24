"""Best possible whole-spectrum agreement under nonincreasing relative size weights."""

import csv
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

root = Path(__file__).resolve().parent
with (
    (root / "channels.csv").open() as source,
    (root / "monotone-limit.csv").open("w") as output,
):
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "wind_ms",
            "best_possible_tv_pp",
            "copy_tv_pp",
            "largest_cdf_deficit_pp",
            "at_channel_um",
        ]
    )
    for wind, group in groupby(csv.DictReader(source), lambda r: r["wind_ms"]):
        rows = list(group)
        d, p, q = (
            np.array([float(r[k]) for r in rows])
            for k in ("channel_um", "early_pct", "measured_pct")
        )
        p, q = p / 100, q / 100
        # Every nonincreasing nonnegative multiplier is a sum of prefix indicators.
        # Normalizing the reweighted spectrum gives a mixture of normalized prefixes.
        ends = np.flatnonzero(p > 0)
        prefix = np.array(
            [p * (np.arange(len(p)) <= j) / p[: j + 1].sum() for j in ends]
        ).T
        n, k = prefix.shape
        eye = np.eye(n)
        result = linprog(
            np.r_[np.zeros(k), np.ones(n)],
            A_ub=np.block([[prefix, -eye], [-prefix, -eye]]),
            b_ub=np.r_[q, -q],
            A_eq=[np.r_[np.ones(k), np.zeros(n)]],
            b_eq=[1],
            bounds=(0, None),
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-10,
                "dual_feasibility_tolerance": 1e-10,
            },
        )
        assert result.success
        projected = prefix @ result.x[:k]
        tv = 50 * np.abs(q - projected).sum()
        assert np.isclose(tv, 50 * result.fun) and np.isclose(
            projected.sum(), 1
        )
        assert np.all(np.diff(projected[ends] / p[ends]) <= 1e-8)
        gap = 100 * np.cumsum(p - q)
        j = gap.argmax()
        writer.writerow(
            [wind, tv, 50 * np.abs(p - q).sum(), max(0, gap[j]), d[j]]
        )
