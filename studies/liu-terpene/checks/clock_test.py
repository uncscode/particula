# %%
"""Test one evaporation clock against Liu Figure 4 without fitting a decay law."""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [("pinene", "ocimene"), ("caryophyllene", "farnesene")]


def save(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def interval(samples, start, stop, dy, dt):
    """Conservative monotone envelope for any time inside [start, stop]."""
    before = [v + dy for t, v in samples if t + dt <= start]
    after = [v - dy for t, v in samples if t - dt >= stop]
    return max([0.0, *after]), min([1.0, *before])


def allowed_clock(cyclic, acyclic, dy, dt):
    """Necessary pairwise constraints for observing the same monotone trajectory."""
    lower, upper = 0.0, np.inf
    lower_pair, upper_pair = "", ""
    for tc, vc in cyclic:
        for ta, va in acyclic:
            if vc - dy > va + dy:
                bound = max(0, tc - dt) / (ta + dt)
                if bound > lower:
                    lower, lower_pair = bound, f"{tc:g}/{ta:g}"
            if vc + dy < va - dy and ta > dt:
                bound = (tc + dt) / (ta - dt)
                if bound < upper:
                    upper, upper_pair = bound, f"{tc:g}/{ta:g}"
    return dict(
        minimum_clock=lower,
        maximum_clock=upper,
        lower_cyclic_acyclic_times=lower_pair,
        upper_cyclic_acyclic_times=upper_pair,
        compatible=lower <= upper,
    )


def analyze(output):
    with (ROOT / "sources/observations.csv").open() as f:
        source = list(csv.DictReader(f))
    ranges, clocks, thresholds = [], [], []
    for c, a in PAIRS:
        for rh in ("dry", "wet"):
            groups = [
                [r for r in source if r["precursor"] == name and r["rh"] == rh]
                for name in (c, a)
            ]
            cyclic, acyclic = [
                np.array([(float(r["time_min"]), float(r["vfr"])) for r in g])
                for g in groups
            ]
            rc, ra = [float(g[0]["density_g_cm3"]) for g in groups]
            dc, da = [float(g[0]["initial_diameter_nm"]) for g in groups]
            # Fractional mass-loss time ~ density * diameter**q, q=1 molecular,
            # q=2 continuum. Bound the whole interpolation and measured +/-0.01 density.
            clock_edges = [
                (rc + s * 0.01) / (ra - s * 0.01) * (dc / da) ** q
                for q in (1, 2)
                for s in (-1, 1)
            ]
            for dy, dt in ((0.0041, 4.6), (0.01, 10.0), (0.025, 10.0)):
                meta = dict(
                    cyclic=c, acyclic=a, rh=rh, reading_vfr=dy, reading_min=dt
                )
                for time_min in (100, 200, 900):
                    lo, hi = interval(
                        cyclic,
                        time_min * min(clock_edges),
                        time_min * max(clock_edges),
                        dy,
                        dt,
                    )
                    alo, ahi = interval(acyclic, time_min, time_min, dy, dt)
                    ranges.append(
                        dict(
                            **meta,
                            time_min=time_min,
                            clock_min=min(clock_edges),
                            clock_max=max(clock_edges),
                            surrogate_low=lo,
                            surrogate_high=hi,
                            acyclic_low=alo,
                            acyclic_high=ahi,
                            difference_low=alo - hi,
                            difference_high=ahi - lo,
                            minimum_relative_density_drift=(alo / hi) ** (1 / 3)
                            - 1,
                        )
                    )
                for cutoff in (0.0, 30.0, 60.0):
                    cg, ag = [g[g[:, 0] >= cutoff] for g in (cyclic, acyclic)]
                    clocks.append(
                        dict(
                            **meta,
                            min_time=cutoff,
                            **allowed_clock(cg, ag, dy, dt),
                        )
                    )
            # How much uniform vertical tolerance per marker is needed to make
            # the unrestricted single-clock contradiction disappear? At a change
            # point two uncertainty intervals first touch. This bounds the
            # discrepancy; it is not an inferred experimental variance.
            for cutoff in (0.0, 30.0):
                cg, ag = [g[g[:, 0] >= cutoff] for g in (cyclic, acyclic)]
                left, right = 0.0, 0.5
                for _ in range(52):
                    mid = (left + right) / 2
                    if allowed_clock(cg, ag, mid, 4.6)["compatible"]:
                        right = mid
                    else:
                        left = mid
                thresholds.append(
                    dict(
                        cyclic=c,
                        acyclic=a,
                        rh=rh,
                        min_time=cutoff,
                        time_tolerance_min=4.6,
                        minimum_vfr_tolerance_per_marker=right,
                    )
                )
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("monotone_ranges", ranges),
        ("constant_clock", clocks),
        ("clock_tolerance", thresholds),
    ]:
        save(output / (name + ".csv"), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent
    )
    analyze(parser.parse_args().output)
