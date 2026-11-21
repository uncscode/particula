# %%
"""Refine the numerical solutions at the extremes of the transfer envelope."""

import dataclasses

import numpy as np
import pandas as pd
from study import FORCING, OUT, case_from, replay, settings

number = pd.read_csv(OUT / "number_check.csv")
curves = {run: (g.time_s, g.conditioned) for run, g in number.groupby("run")}
count_fit = pd.read_csv(OUT / "number_calibration.csv").iloc[0]
envelope = pd.read_csv(OUT / "run5_fit_envelope.csv").sort_values("model_peak")
out = []
for label, row, number_forcing in [
    ("lowest_run5_peak", envelope.iloc[0], None),
    ("highest_run5_peak", envelope.iloc[-1], None),
    ("count_constrained", count_fit, curves),
]:
    case = dataclasses.replace(
        case_from(row, 5), wall_share=row.get("wall_share", 1)
    )
    config = settings(row)
    fine_t, fine, checks = replay(
        [case],
        FORCING,
        dt=3,
        interval=180,
        number_forcing=number_forcing,
        **config,
    )
    t, coarse, _ = replay(
        [case], FORCING, number_forcing=number_forcing, **config
    )
    out.append(
        dict(
            test=label,
            coarse_peak=coarse[:, 0, 0].max(),
            refined_peak=fine[:, 0, 0].max(),
            max_absolute_difference=np.max(
                abs(np.interp(t, fine_t, fine[:, 0, 0]) - coarse[:, 0, 0])
            ),
            **checks,
        )
    )
pd.DataFrame(out).to_csv(OUT / "transfer_refinement.csv", index=False)
