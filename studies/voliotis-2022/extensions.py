# %% [markdown]
# Freeze the original fit. Bound the missing final hour of precursor forcing
# with two continuations, and test what Table 1's injection uncertainty permits.
# %%
"""Bound injection error, vapor loss and the unmeasured final gas hour."""

import dataclasses
import json

import numpy as np
import pandas as pd
from study import FORCING, OUT, case_from, metrics, replay, series, settings

fit = json.loads((OUT / "frozen_run4_fit.json").read_text())
params = settings(fit)
cases = [case_from(fit, run) for run in (4, 5)]
out = []
records = []
for scenario in ("stop_oxidation", "continue_final_rate"):
    extended = {}
    for run, (t, f) in FORCING.items():
        hazard = np.log(f[-2] / f[-1]) / (t[-1] - t[-2])
        end_fraction = (
            f[-1]
            if scenario == "stop_oxidation"
            else f[-1] * np.exp(-hazard * 3600)
        )
        extended[run] = (
            np.r_[t, np.arange(18360, 21601, 360)],
            np.r_[
                f,
                np.exp(np.linspace(np.log(f[-1]), np.log(end_fraction), 11))[
                    1:
                ],
            ],
        )
    ts, h, check = replay(cases, extended, end=21600, **params)
    for i, c in enumerate(cases):
        t, y = series(c.run, "soa_ug_m3")
        use = (t > 18000) & (t <= 21600)
        pred = np.interp(t[use], ts, h[:, i, 0])
        out.append(
            dict(
                scenario=scenario,
                run=c.run,
                pred_6h=h[-1, i, 0],
                obs_6h=y[-1],
                rmse_final_hour=np.mean((pred - y[use]) ** 2) ** 0.5,
                **check,
            )
        )
        for j, sec in enumerate(ts):
            records.append(
                dict(scenario=scenario, run=c.run, time_s=sec, soa=h[j, i, 0])
            )
pd.DataFrame(out).to_csv(OUT / "unmeasured_gas_tail.csv", index=False)
pd.DataFrame(records).to_csv(OUT / "tail_trajectories.csv", index=False)

# A kinetic identity: scaling the run-4 injection by q and dividing its
# fitted branch by q preserves the complete condensable production history.
# Transfer the re-expressed branch to run 5 at the opposite injection limit.
scenarios = [
    (
        "low_relative_injection",
        dataclasses.replace(
            cases[1], injection=0.85, fraction=cases[1].fraction / 1.15
        ),
    ),
    (
        "high_relative_injection",
        dataclasses.replace(
            cases[1], injection=1.15, fraction=cases[1].fraction / 0.85
        ),
    ),
    ("add_vapor_wall_0.6", dataclasses.replace(cases[1], vapor_wall_h=0.6)),
    ("add_vapor_wall_1.8", dataclasses.replace(cases[1], vapor_wall_h=1.8)),
    (
        "check_run4_rescaling",
        dataclasses.replace(
            cases[0], injection=0.85, fraction=cases[0].fraction / 0.85
        ),
    ),
]
t, h, checks = replay([c for _, c in scenarios], FORCING, **params)
pd.DataFrame(
    [
        dict(
            scenario=name, run=c.run, **metrics(t, h[:, i, 0], c.run), **checks
        )
        for i, (name, c) in enumerate(scenarios)
    ]
).to_csv(OUT / "injection_and_walls.csv", index=False)
