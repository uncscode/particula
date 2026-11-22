"""Measure how much candidate mass transfer can hide in the joint score."""

import json

import numpy as np
import pandas as pd
from baseline_replay import HERE, Model, run
from growth import BANK, CAP, initial


def main():
    """Freeze the selected parent and vary only each fine finalist's rate."""
    bank_ids = {channel.key: i for i, channel in enumerate(BANK)}
    fixed = json.loads((HERE / "size_inputs.json").read_text())["cases"][
        "selected"
    ]
    best_ids = [bank_ids[r["key"]] for r in fixed["channels"]]
    base, _ = initial()
    base[:] = 0
    base[:16] = fixed["x"][:16]
    base[16 + np.array(best_ids)] = fixed["x"][16:]
    doses = np.array(fixed["dose"])[:, None]
    model = Model(BANK)
    rows = []
    dt = json.loads((HERE / "diagnostics.json").read_text())["selected"][
        "trajectory_step_seconds"
    ]
    sub = round(60 / dt)
    keys = ("a_return", "accrete_3_3", "CC_29_24")
    for key in keys:
        idx = bank_ids[key]
        active = sorted(set(best_ids) | {idx})
        rates = np.unique([0.0, 0.01, 0.1, 0.5, 1.0, CAP[idx], base[16 + idx]])
        points = np.tile(base, (len(rates), 1))
        points[:, 16 + idx] = rates
        forward, errors, _, gross = run(
            model, points, active, sub, doses, diagnostics=1
        )
        reverse, _, _, _ = run(model, points, active, sub, doses, reverse=True)
        for i, rate in enumerate(rates):
            for j, (experiment, _, _, observed, _) in enumerate(model.runs):
                rows.append(
                    dict(
                        key=key,
                        step_s=dt,
                        fitted_rate=base[16 + idx],
                        rate=rate,
                        run=experiment,
                        forward_joint=forward[i],
                        reverse_joint=reverse[i],
                        mass_rmse=errors[i, j, 0],
                        number_rmse=errors[i, j, 1],
                        observed_drop=max(observed) - observed[-1],
                        model_drop=errors[i, j, 3] - errors[i, j, 4],
                        gross_ug_m3=gross[i, j, idx],
                    )
                )
    pd.DataFrame(rows).to_csv(HERE / "flux_response.csv", index=False)


if __name__ == "__main__":
    main()
