"""Compare isomer timing and a loading-calibrated one-surrogate model."""

import itertools
from pathlib import Path

import numpy as np
import warp as wp
from scripts.pinic_isomers.physics import (
    PINIC,
    TOTAL,
    effective_cstar,
    equilibrium,
    trajectory,
)


def first_crossing(time, trace, smooth):
    """Interpolate the first post-start 50% crossing using centered averages."""
    kernel = np.full(smooth, 1 / smooth)
    avg = np.convolve(trace, kernel, mode="valid")
    x = time[smooth // 2 : len(time) - smooth // 2]
    i = np.flatnonzero((x >= 0) & (avg >= 0.5))[0]
    return float(
        x[i - 1]
        + (x[i] - x[i - 1]) * (0.5 - avg[i - 1]) / (avg[i] - avg[i - 1])
    )


def simulate():
    """Run paired resolved and surrogate CUDA trajectories for every mixture."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/data/pinic-isomers/traces.csv"
    )
    traces = np.loadtxt(path, delimiter=",", skiprows=1)
    timings = {
        str(width): [
            first_crossing(traces[:, 0], col, width) for col in traces[:, 1:].T
        ]
        for width in (1, 3, 5)
    }
    seeds = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    times = np.array([60.0, 600.0, 3600.0, 21600.0, 86400.0])
    cases = []
    max_fixed_error = 0.0
    max_budget = 0.0
    max_null = 0.0
    max_refined = 0.0
    for ratio, fraction in itertools.product(
        (0.01, 0.1, 1.0, 10.0, 100.0), (0.1, 0.5, 0.9)
    ):
        totals = TOTAL * np.array([fraction, 1 - fraction])
        cstar = PINIC * np.array([1.0, ratio])
        single = np.array([effective_cstar(totals, cstar)])
        resolved, b1 = trajectory(seeds, totals, cstar, times, 2e-5)
        lumped, b2 = trajectory(seeds, np.array([TOTAL]), single, times, 2e-5)
        max_budget = max(max_budget, b1, b2)
        resolved = np.asarray(resolved)
        lumped = np.asarray(lumped)[:, :, 0]
        for end, inputs, vp in [
            (resolved[-1].sum(axis=1), totals, cstar),
            (lumped[-1], np.array([TOTAL]), single),
        ]:
            fixed = np.array([equilibrium(seed, inputs, vp) for seed in seeds])
            max_fixed_error = max(
                max_fixed_error, float(np.max(abs(end - fixed)))
            )
        if ratio == 1:
            max_null = max(
                max_null, float(np.max(abs(lumped - resolved.sum(axis=2))))
            )
        if fraction == 0.5 and ratio in (0.01, 100.0):
            refined, b3 = trajectory(seeds, totals, cstar, times, 2e-6)
            max_refined = max(
                max_refined, float(np.max(abs(resolved - np.asarray(refined))))
            )
            max_budget = max(max_budget, b3)
        np.testing.assert_allclose(
            lumped[-1, 4], resolved[-1, 4].sum(), rtol=1e-5
        )
        cases.append(
            dict(
                ratio=ratio,
                pinic_fraction=fraction,
                single_cstar=single.item(),
                resolved=resolved.tolist(),
                lumped=lumped.tolist(),
            )
        )
    if max_fixed_error > 2e-5 or max_null > 2e-4 or max_refined > 3e-4:
        raise ValueError(
            f"Convergence failed: fixed {max_fixed_error}, null {max_null}, "
            f"refined {max_refined}"
        )
    return dict(
        cases=cases,
        seeds=seeds.tolist(),
        times_s=times.tolist(),
        pinic_cstar=PINIC,
        traces=traces.tolist(),
        half_times=timings,
        max_fixed_error=max_fixed_error,
        max_budget=max_budget,
        max_null=max_null,
        max_refined=max_refined,
        device=wp.get_device("cuda:0").name,
    )
