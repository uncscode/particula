# %%
"""Propagate bounded chemistry and wall mixing for α-pinene."""

import json

import numpy as np
import pandas as pd
from engine import RESULT, Engine
from optimize import PHYSICAL_NAMES
from scipy.stats import qmc
from uncertainty import SEEDS, shares
from uncertainty_profile import DEST, anchors
from wall_fit import ACTIVE, OUT, RUNS, WallObjective, simulate


def main():
    """Cross chemistry, vapor loss and uptake with particle deposition."""
    archived = json.loads((OUT / "restarts.json").read_text())
    dark = json.loads((OUT / "dark_restarts.json").read_text())["alpha_pinene"]
    fits = [r for r in archived if r["model"] == "eddy_brownian"] + dark
    corners = np.array([r["coords"][:6] for r in fits])
    lo, hi = corners.min(0), corners.max(0)
    u = np.vstack(
        [qmc.Sobol(6, seed=s, scramble=True).random_base2(7) for s in SEEDS]
    )
    sampled = lo + u * (hi - lo)
    coords = np.vstack((sampled, corners))
    physical = coords.copy()
    physical[:, 1:4] = 10 ** physical[:, 1:4]
    design = pd.DataFrame(physical, columns=[PHYSICAL_NAMES[i] for i in ACTIVE])
    design["source"] = [f"sobol_{s}" for s in SEEDS for _ in range(128)] + [
        "anchor"
    ] * len(fits)
    design.to_csv(DEST / "alpha_design.csv", index_label="scenario")
    original = np.array(
        json.loads((RESULT / "single_nox.json").read_text())["physical"]
    )
    engine, objective = Engine(3), WallObjective()
    _, levels, *_ = anchors()
    forward, summary = [], []
    for e in levels:
        pars = np.column_stack((coords, np.full(len(coords), np.log10(e))))
        mass, number = simulate(engine, pars, "eddy_brownian", original)
        me, ne = objective.errors(mass, number)
        score = np.maximum(me, ne).max(1)
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite alpha-pinene score")
        forward.append(
            np.stack(
                (mass[: len(u), :4].max(2), number[: len(u), :4, -1]), axis=-1
            )
        )
        best = score.argmin()
        summary.append(
            dict(
                eddy_s_inv=e,
                best_joint=score[best],
                scenario=best,
                worst_mass=me[best].max(),
                worst_count=ne[best].max(),
                best_sampled=score[: len(u)].min(),
            )
        )
    pd.DataFrame(summary).to_csv(DEST / "alpha_profile.csv", index=False)
    y = np.array(forward)
    fractions = shares(y)
    rows = []
    for i, r in enumerate(RUNS):
        for j, outcome in enumerate(
            ("peak_soa_ug_m3_grid", "five_hour_number_fraction")
        ):
            rows.append(
                dict(
                    run=r,
                    outcome=outcome,
                    design_min=y[:, :, i, j].min(),
                    design_max=y[:, :, i, j].max(),
                    chemistry_uptake_vapor_loss_fraction=fractions[0, i, j],
                    wall_fraction=fractions[1, i, j],
                    interaction_fraction=fractions[2, i, j],
                )
            )
    pd.DataFrame(rows).to_csv(DEST / "alpha_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
