# %%
"""Bound the fraction of cyclic loss reproduced by acyclic SOA without a kinetic fit."""

import argparse
import csv
from pathlib import Path

import numpy as np
from clock_test import PAIRS, ROOT, allowed_clock, interval, save


def clock_grid(cyclic, acyclic, ends, dt):
    """Inequality constraints change only at a relative ordering of sample times."""
    edges = set(ends)
    for tc, _ in cyclic:
        for ta, _ in acyclic:
            for numerator, denominator in (
                (tc + dt, ta - dt),
                (tc - dt, ta + dt),
            ):
                if (
                    denominator > 0
                    and ends[0] < numerator / denominator < ends[1]
                ):
                    edges.add(numerator / denominator)
    edges = sorted(edges)
    return [*edges, *((a + b) / 2 for a, b in zip(edges, edges[1:]))]


def loss_range(cyclic, training, clock, dt, dy):
    """Necessary bounds on b in V_a=1-b(1-V_c); interval errors on both axes."""
    lower, upper = 0.0, 1.0
    for ta, va in training:
        lo, hi = interval(
            cyclic, max(0.0, ta - dt) * clock, (ta + dt) * clock, dy, dt
        )
        if lo < 1:
            lower = max(lower, (1 - va - dy) / (1 - lo))
        if hi < 1:
            upper = min(upper, (1 - va + dy) / (1 - hi))
    return lower, upper


def analyze(output):
    with (ROOT / "sources/observations.csv").open() as f:
        source = list(csv.DictReader(f))
    summaries, predictions, actual = [], [], []
    for c, a in PAIRS:
        for rh in ("dry", "wet"):
            sets = [
                [r for r in source if r["precursor"] == k and r["rh"] == rh]
                for k in (c, a)
            ]
            cyclic, acyclic = [
                np.array(
                    [(float(r["time_min"]), float(r["vfr"])) for r in rows]
                )
                for rows in sets
            ]
            dc, da = [float(rows[0]["initial_diameter_nm"]) for rows in sets]
            rc, ra = [float(rows[0]["density_g_cm3"]) for rows in sets]
            for label, qrange in [
                ("gas_transport", (1, 2)),
                ("size_independent_to_continuum", (0, 2)),
            ]:
                speeds = [
                    (rc + s * 0.01) / (ra - s * 0.01) * (dc / da) ** q
                    for s in (1, -1)
                    for q in qrange
                ]
                ends = (min(speeds), max(speeds))
                for dy, dt in [(0.0041, 4.6), (0.01, 10.0), (0.025, 10.0)]:
                    meta = dict(
                        cyclic=c,
                        acyclic=a,
                        rh=rh,
                        clock_family=label,
                        dy=dy,
                        dt=dt,
                    )
                    for ta, va in acyclic:
                        lo, hi = interval(
                            cyclic,
                            max(0.0, ta - dt) * ends[0],
                            (ta + dt) * ends[1],
                            dy,
                            dt,
                        )
                        actual.append(
                            dict(
                                **meta,
                                time_min=ta,
                                acyclic_vfr=va,
                                cyclic_low=lo,
                                cyclic_high=hi,
                                residual_min=va - dy - hi,
                                residual_max=va + dy - lo,
                            )
                        )
                    grid = clock_grid(cyclic, acyclic, ends, dt)
                    training = acyclic[acyclic[:, 0] <= 300]
                    held = acyclic[acyclic[:, 0] > 900]
                    rate = allowed_clock(cyclic, training, dy, dt)
                    for scope, ag in [("daytime", training), ("all", acyclic)]:
                        fits = []
                        for clock in grid:
                            lo, hi = loss_range(cyclic, ag, clock, dt, dy)
                            if lo <= hi:
                                fits.append((clock, lo, hi))
                        summaries.append(
                            dict(
                                **meta,
                                scope=scope,
                                pool_compatible=bool(fits),
                                extra_fraction_low=min(
                                    (1 - hi for _, _, hi in fits), default=""
                                ),
                                extra_fraction_high=max(
                                    (1 - lo for _, lo, _ in fits), default=""
                                ),
                            )
                        )
                        if scope != "daytime":
                            continue
                        for ta, va in held:
                            bounds = []
                            for clock, blo, bhi in fits:
                                lo, hi = interval(
                                    cyclic,
                                    (ta - dt) * clock,
                                    (ta + dt) * clock,
                                    dy,
                                    dt,
                                )
                                bounds.append(
                                    (1 - bhi * (1 - lo), 1 - blo * (1 - hi))
                                )
                            if rate["compatible"]:
                                rtlo, rthi = interval(
                                    cyclic,
                                    (ta - dt) * rate["minimum_clock"],
                                    (ta + dt) * rate["maximum_clock"],
                                    dy,
                                    dt,
                                )
                            else:
                                rtlo, rthi = None, None
                            predictions.append(
                                dict(
                                    **meta,
                                    time_min=ta,
                                    observed=va,
                                    retained_pool_low=min(
                                        (b[0] for b in bounds), default=None
                                    ),
                                    retained_pool_high=max(
                                        (b[1] for b in bounds), default=None
                                    ),
                                    uniform_rate_low=rtlo,
                                    uniform_rate_high=rthi,
                                )
                            )
    for name, rows in [
        ("pool_bounds", summaries),
        ("pool_holdout", predictions),
        ("marker_bounds", actual),
    ]:
        save(output / (name + ".csv"), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent
    )
    where = parser.parse_args().output
    where.mkdir(parents=True, exist_ok=True)
    analyze(where)
