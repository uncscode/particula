# %% [markdown]
# A second aerosol measurement tests the particle removal inferred from just
# the organic mass curve. Treat the measured count decline as a constraint,
# and compare removal of material with consolidation into larger particles.
# %%
"""Condition chamber simulations on independent particle number measurements."""

import itertools
import json

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression
from study import (
    FORCING,
    HERE,
    HORIZON,
    OUT,
    Case,
    metrics,
    replay,
    settings,
)

df = pd.read_csv(HERE / "data/context.csv")
fit = json.loads((OUT / "frozen_run4_fit.json").read_text())
counts, diagnostics = {}, []
for run, instrument, column in [
    (4, "WCPC", "total_NumberConc"),
    (5, "DMPS", "Total_number"),
    (1, "WCPC", "total_NumberConc"),
    (6, "WCPC", "total_NumberConc"),
]:
    subset = df[
        (df.run == run) & (df.instrument == instrument) & (df.channel == column)
    ].sort_values("time_s")
    t, n = subset.time_s.to_numpy(), subset.value.to_numpy()
    n0 = n[np.argmin(abs(t))]
    grid = np.arange(0, HORIZON + 1, 360)
    raw = np.interp(grid, t, n) / n0
    conditioned = isotonic_regression(np.clip(raw, 0, 1), increasing=False).x
    counts[run] = (grid, conditioned)
    for i, sec in enumerate(grid):
        diagnostics.append(
            dict(
                run=run,
                time_s=sec,
                observed_interpolated=raw[i],
                conditioned=conditioned[i],
                mass_only=np.exp(-fit["particle_wall_h"] * sec / 3600),
            )
        )
pd.DataFrame(diagnostics).to_csv(OUT / "number_check.csv", index=False)

cases = [
    Case(4, f, wall, 0.0, lag, wall_share=share)
    for f, wall, lag, share in itertools.product(
        np.arange(0.1, 0.701, 0.1),
        (0.0, 0.6, 1.8, 4.5),
        (5.0, 25.0),
        (0.0, 0.5, 1.0),
    )
]
rows = []
for cstar, accom in itertools.product((1.0, 10.0, 100.0), (0.03, 0.3)):
    params = dict(cstar=cstar, accommodation=accom, seed_access=1.0)
    t, h, checks = replay(cases, FORCING, number_forcing=counts, **params)
    for i, c in enumerate(cases):
        rows.append(
            dict(
                fraction=c.fraction,
                vapor_wall_h=c.vapor_wall_h,
                particle_wall_h=0.0,
                lag_min=c.lag_min,
                wall_share=c.wall_share,
                **params,
                **checks,
                **metrics(t, h[:, i, 0], 4),
            )
        )
    print("Count-conditioned training", cstar, accom, flush=True)
grid = pd.DataFrame(rows).sort_values("rmse", kind="stable")
grid.to_csv(OUT / "number_calibration.csv", index=False)
best = grid.iloc[0]
rows = []
traces = []
for label, row in [("best", best)] + [
    (f"share_{share:g}", g.iloc[0]) for share, g in grid.groupby("wall_share")
]:
    runs = [
        Case(
            r,
            row.fraction,
            row.vapor_wall_h,
            0.0,
            row.lag_min,
            wall_share=row.wall_share,
        )
        for r in (4, 5, 1, 6)
    ]
    t, h, checks = replay(runs, FORCING, number_forcing=counts, **settings(row))
    for i, c in enumerate(runs):
        rows.append(
            dict(
                test=label,
                run=c.run,
                wall_share=c.wall_share,
                **metrics(t, h[:, i, 0], c.run),
                **checks,
            )
        )
        for j, sec in enumerate(t):
            traces.append(
                dict(test=label, run=c.run, time_s=sec, soa=h[j, i, 0])
            )
pd.DataFrame(rows).to_csv(OUT / "number_transfer.csv", index=False)
pd.DataFrame(traces).to_csv(OUT / "number_trajectories.csv", index=False)
