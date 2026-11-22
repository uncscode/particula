"""Align simulated mass loss with each experiment's observed peak time."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    """Measure both independent and observed-peak-aligned declines."""
    data = pd.read_csv(HERE / "instrument_fits.csv.gz")
    rows = []
    for (network, run), group in data.query("observable=='mass'").groupby(
        ["network", "run"]
    ):
        g = group.sort_values("time_h")
        i = g.loc[g.value.idxmax()]
        j = g.loc[g.prediction.idxmax()]
        end = g.iloc[-1]
        rows.append(
            dict(
                network=network,
                run=run,
                observed_peak_h=i.time_h,
                predicted_peak_h=j.time_h,
                last_sample_h=end.time_h,
                observed_drop_ug_m3=i.value - end.value,
                model_own_peak_drop_ug_m3=j.prediction - end.prediction,
                model_observed_peak_drop_ug_m3=i.prediction - end.prediction,
                observed_peak_bias_ug_m3=i.prediction - i.value,
                last_bias_ug_m3=end.prediction - end.value,
                modeled_wall_increment_ug_m3=end.deposited_ug_m3
                - i.deposited_ug_m3,
                forcing_increment_ug_m3=end.consumed_ug_m3 - i.consumed_ug_m3,
            )
        )
    result = pd.DataFrame(rows)
    dt = json.loads((HERE / "diagnostics.json").read_text())["selected"][
        "trajectory_step_seconds"
    ]
    audit = pd.read_csv(HERE / "scorecard.csv")
    audit = audit[audit.step == f"{dt:g}s"]
    aligned = result.merge(audit, on=["network", "run"], validate="one_to_one")
    np.testing.assert_allclose(
        aligned.model_own_peak_drop_ug_m3, aligned["drop"], atol=1e-10
    )
    np.testing.assert_allclose(
        aligned.observed_drop_ug_m3, aligned.observed_drop, atol=1e-10
    )
    result.to_csv(HERE / "peak_timing.csv", index=False)


if __name__ == "__main__":
    main()
