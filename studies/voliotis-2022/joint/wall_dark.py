# %%
"""Refit chemistry with wall transport constrained by dark seed experiments."""

import json

import numpy as np
import pandas as pd
from engine import HERE, RESULT, Engine
from wall_campaign import Campaign, search
from wall_fit import OUT, RUNS, WallObjective, fit, simulate


def main():
    """Use the external apparent eddy rate and retain each run's residuals."""
    reference = json.loads((HERE / "wall_data/wall_fit.json").read_text())
    eddy = reference["fits"]["all_readable_log"]["eddy_s-1"]
    original = json.loads((RESULT / "single_nox.json").read_text())["physical"]
    records = json.loads((OUT / "restarts.json").read_text())
    start = min(
        (
            r
            for r in records
            if r["model"] == "eddy_brownian" and r["objective"] == "joint"
        ),
        key=lambda r: r["score"],
    )["coords"]
    scorer = WallObjective()
    fits = fit(
        "eddy_brownian",
        "joint",
        np.array(original),
        [Engine(18), Engine(3)],
        scorer,
        np.array(start),
        fixed_eddy=eddy,
    )
    best = min(fits, key=lambda r: r["score"])
    m, n = scorer.errors(
        *simulate(Engine(3), best["coords"], "eddy_brownian", original)
    )
    rows = [
        dict(
            study="alpha_pinene",
            run=r,
            eddy_s=eddy,
            mass_nrmse=m[0, i],
            count_nrmse=n[0, i],
        )
        for i, r in enumerate(RUNS)
    ]
    archive = {"alpha_pinene": fits}
    campaign = Campaign()
    base = pd.read_csv(RESULT / "campaign_dose_scores.csv")
    row = base[(base.family == "parent_lags") & (base.stage == "joint")].iloc[0]
    start = np.r_[json.loads(row.lag_h), json.loads(row.yields), eddy]
    fits = search(campaign, True, "joint", start, fixed_eddy=eddy)
    best = min(fits, key=lambda r: r["score"])
    _, err, _ = campaign.evaluate(
        best["physical"], True, "joint", np.linspace(0.85, 1.15, 61), sub=2
    )
    rows.extend(
        dict(
            study="campaign",
            run=r[0],
            eddy_s=eddy,
            mass_nrmse=err[0, i, 0],
            count_nrmse=err[0, i, 1],
        )
        for i, r in enumerate(campaign.runs)
    )
    archive["campaign"] = fits
    pd.DataFrame(rows).to_csv(OUT / "dark_constrained.csv", index=False)
    (OUT / "dark_restarts.json").write_text(
        json.dumps(archive, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
