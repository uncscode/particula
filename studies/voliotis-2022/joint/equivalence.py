# %%
"""Find NOx slopes and doses that reproduce exactly the same product source."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from engine import NOX, ORDER, RESULT
from scipy.optimize import linprog
from scipy.special import expit, logit


def equivalent_sources():
    """Bound slopes that preserve fitted source strengths in four runs."""
    base = json.loads((RESULT / "single_nox.json").read_text())["physical"]
    runs = ORDER[:4]
    nox = np.array([NOX[r] for r in runs])
    z = np.log(nox / 30.0)
    target = expit(logit(base[0]) + base[16] * z)
    # Holding dose_i * branch_i constant preserves each whole trajectory.
    # Dose_i in [0.85, 1.15] sets lower and upper bounds on logit(branch_i).
    design = np.column_stack((np.ones(len(z)), z))
    lower = logit(target / 1.15)
    upper = logit(np.minimum(target / 0.85, 1 - np.finfo(float).eps))
    matrix = np.vstack([design, -design])
    rhs = np.r_[upper, -lower]
    bounds = [(logit(0.05), logit(0.85)), (-3, 3)]
    rows = []
    for label, sign in (("lowest", 1), ("nominal", 0), ("highest", -1)):
        if sign:
            opt = linprog([0, sign], A_ub=matrix, b_ub=rhs, bounds=bounds)
            if not opt.success:
                raise RuntimeError(opt.message)
            intercept, beta = opt.x
        else:
            intercept, beta = logit(base[0]), base[16]
        branches = expit(intercept + beta * z)
        doses = target / branches
        if np.any((doses < 0.85 - 1e-10) | (doses > 1.15 + 1e-10)):
            raise ValueError("Dose outside the uncertainty envelope")
        for run, branch, dose, strength in zip(
            runs, branches, doses, target, strict=True
        ):
            rows.append(
                dict(
                    solution=label,
                    run=run,
                    initial_NOx_ppb=NOX[run],
                    slope=beta,
                    intercept_at_30ppb=expit(intercept),
                    branch=branch,
                    dose_multiplier=dose,
                    effective_source=strength,
                )
            )
    return pd.DataFrame(rows), z, lower, upper


if __name__ == "__main__":
    df, z, lower, upper = equivalent_sources()
    df.to_csv(RESULT / "equivalent_sources.csv", index=False)
    solutions = df.drop_duplicates("solution").set_index("solution")
    b0, b1 = solutions.loc[["lowest", "highest"], "slope"]
    slopes = np.linspace(b0, b1, 300)
    ceilings = np.minimum(logit(0.85), (upper - slopes[:, None] * z).min(1))
    floors = np.maximum(logit(0.05), (lower - slopes[:, None] * z).max(1))
    fig, ax = plt.subplots(figsize=(6.2, 3.6), layout="constrained")
    ax.fill_between(
        slopes,
        expit(floors),
        expit(ceilings),
        color="#91c4bd",
        label="Exactly equivalent with doses within ±15%",
    )
    fit = solutions.loc["nominal"]
    ax.scatter(
        fit.slope,
        fit.intercept_at_30ppb,
        color="#332f52",
        label="Nominal doses",
    )
    ax.set(
        xlabel="Slope of condensable branching versus initial NOₓ (β)",
        ylabel="Condensable fraction at 30 ppb NOₓ",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(loc="outside lower center", frameon=False, fontsize=9)
    fig.savefig(RESULT / "equivalent-sources.png", dpi=190)
    plt.close(fig)
