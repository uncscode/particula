# %%
"""Fit reduced product chemistry across the measured multi-precursor runs."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression, linprog, minimize
from scipy.signal import lfilter
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
SPECIES = ("alpha_pinene", "o_cresol", "isoprene")
MW = np.array([136.238, 108.14, 68.117])  # g mol⁻¹
PPB_TO_UG_M3 = MW * 101325 / (8.314462618 * 298.15) * 1e-3
# At 298.15 K and 1 atm this is molecular weight divided by 24.465 L/mol.
STEP_S = 60


def inputs(knot_seconds=1200):
    """Match original AMS samples with full measured parent histories."""
    primary = pd.read_csv(HERE.parent / "data/observations.csv").rename(
        columns={"kind": "variable"}
    )
    primary["variable"] = primary.variable.replace(
        {"apinene_fraction": "alpha_pinene_fraction"}
    )
    additional = pd.read_csv(HERE / "data/campaign_observations.csv")
    obs = pd.concat([primary, additional[~additional.run.isin(primary.run)]])
    conditions = pd.read_csv(HERE / "data/campaign_peaks.csv").set_index("run")
    runs = []
    for run, df in obs.groupby("run", sort=True):
        doses = conditions.loc[run, [s + "_ppb" for s in SPECIES]].astype(float)
        vocs = [
            df[df.variable == s + "_fraction"].sort_values("time_s")
            for s in SPECIES
        ]
        ams = df[df.variable == "soa_ug_m3"].sort_values("time_s")
        if ams.empty or any(
            v.empty for d, v in zip(doses, vocs, strict=True) if d
        ):
            continue
        last = min(
            [18000]
            + [v.time_s.max() for d, v in zip(doses, vocs, strict=True) if d]
        )
        steps = np.arange(0, last + 1, STEP_S)  # cover the gas window
        if len(steps) < 61:
            continue
        grid = np.unique(
            np.r_[np.arange(0, steps[-1], knot_seconds), steps[-1]]
        )
        ams = ams[(ams.time_s >= 0) & (ams.time_s <= steps[-1])]
        mass = np.zeros((len(steps) - 1, 3))
        for i, (d, v) in enumerate(zip(doses, vocs, strict=True)):
            if d:
                trace = np.clip(np.interp(grid, v.time_s, v.value), 0, 1)
                trace[0] = 1.0
                f = isotonic_regression(trace, increasing=False).x
                mass[:, i] = (
                    -np.diff(np.interp(steps, grid, f)) * d * PPB_TO_UG_M3[i]
                )
        total = mass.sum(axis=1, keepdims=True)
        own = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
        # The two channels sum to each parent's loss at every time step.
        mix = np.column_stack((mass * own, mass * (1 - own)))
        runs.append(
            (run, steps, ams.time_s.to_numpy(), ams.value.to_numpy(), mix)
        )
    return runs


def basis(run_data, lag_h, removal_h):
    """Integrate intermediate and suspended-product reservoirs."""
    _, steps, times, _, mass = run_data
    dt = STEP_S / 3600
    k, w = 1 / lag_h, removal_h
    a, b = np.exp(-dt * np.array([k, w]))
    coupling = k * a * (-np.expm1(-(w - k) * dt) / (w - k) if w != k else dt)
    source = (-np.expm1(-w * dt) / w - coupling / k) / dt
    # The piecewise-constant source has an exact two-reservoir recurrence.
    into_i = -np.expm1(-k * dt) / (k * dt)
    intermediate = lfilter([into_i], [1, -a], mass, axis=0)
    previous = np.vstack([np.zeros(mass.shape[1]), intermediate[:-1]])
    aerosol = np.vstack(
        [
            np.zeros(mass.shape[1]),
            lfilter([1], [1, -b], coupling * previous + source * mass, axis=0),
        ]
    )
    return np.column_stack([np.interp(times, steps, y) for y in aerosol.T])


def design_matrices(runs, lag_h, removal_h, family):
    """Build the yield-response matrices for the selected kinetic rates."""
    full = []
    for r in runs:
        if np.ndim(lag_h) == 0:
            full.append(basis(r, lag_h, removal_h))
        else:
            m = np.zeros((len(r[2]), 6))
            for j, tau in enumerate(lag_h):
                ix = [j, j + 3]
                if r[4][:, ix].any():
                    m[:, ix] = basis((*r[:4], r[4][:, ix]), tau, removal_h)
            full.append(m)
    return [m[:, :3] + m[:, 3:] for m in full] if family == "additive" else full


def fit(matrices, targets, scales):
    """Solve the joint convex minimax problem for bounded mass yields."""
    mats = [m / s for m, s in zip(matrices, scales, strict=True)]
    obs = [y / s for y, s in zip(targets, scales, strict=True)]
    gram = np.stack([x.T @ x / len(x) for x in mats])
    cross = np.stack([x.T @ y / len(y) for x, y in zip(mats, obs, strict=True)])
    norm = np.array([np.mean(y**2) for y in obs])
    # Rescale each yield so the isoprene near-zero controls remain conditioned.
    scale = 1 / np.maximum(np.sqrt(gram.diagonal(axis1=1, axis2=2).max(0)), 1)
    gram *= scale[None, :, None] * scale[None, None, :]
    cross *= scale

    def constraints(v):
        y = v[:-1]
        return v[-1] - (
            np.einsum("i,rij,j->r", y, gram, y) - 2 * cross @ y + norm
        )

    def gradient(v):
        deriv = -2 * (gram @ v[:-1] - cross)
        return np.column_stack((deriv, np.ones(len(deriv))))

    dim = len(scale)
    cost = np.r_[np.zeros(dim), 1.0]
    start = np.r_[np.zeros(dim), norm.max() + 0.01]
    result = minimize(
        lambda v: v[-1],
        start,
        jac=lambda _: cost,
        method="SLSQP",
        bounds=[(0, 2 / s) for s in scale] + [(0, None)],
        constraints=[dict(type="ineq", fun=constraints, jac=gradient)],
        options={"ftol": 1e-11, "maxiter": 300},
    )
    # Every tangent to a convex squared error underestimates it globally.
    # The minimax linear program on accumulated tangents certifies the gap.
    bounds = [(0, 2 / s) for s in scale] + [(0, None)]
    point = result.x[:-1]
    best_point = point.copy()
    best_loss = np.inf
    planes = []
    intercepts = []
    for _ in range(150):
        losses = (
            np.einsum("i,rij,j->r", point, gram, point)
            - 2 * cross @ point
            + norm
        )
        grads = 2 * (gram @ point - cross)
        if losses.max() < best_loss:
            best_loss = losses.max()
            best_point = point.copy()
        planes.append(np.column_stack((grads, -np.ones(len(grads)))))
        intercepts.extend(grads @ point - losses)
        lower = linprog(
            cost, A_ub=np.vstack(planes), b_ub=intercepts, bounds=bounds
        )
        if not lower.success:
            raise ArithmeticError(lower.message)
        if best_loss - lower.fun < 2e-7:
            return best_point * scale
        point = lower.x[:-1]
    raise ArithmeticError("Could not certify the inner yield optimum")


if __name__ == "__main__":
    runs = inputs()
    if len(runs) <= 4:
        raise ValueError("Additional measured precursor systems required")
    scores, curves, designs = [], [], []
    targets = [r[3] for r in runs]
    scales = [max(1.0, y.max()) for y in targets]
    best = {}

    def evaluate(pair, family, stage):
        """Fit yields at these kinetic rates and retain the lowest objective."""
        removal_h = pair[-1]
        lag_h = (
            float(pair[0]) if len(pair) == 2 else [float(v) for v in pair[:-1]]
        )
        mats = design_matrices(runs, lag_h, removal_h, family)
        yields = fit(mats, targets, scales)
        pred = [m @ yields for m in mats]
        worst = max(
            np.mean((p - y) ** 2) ** 0.5 / s
            for p, y, s in zip(pred, targets, scales, strict=True)
        )
        designs.append(
            dict(
                family=family,
                stage=stage,
                lag_h=lag_h,
                removal_h=removal_h,
                worst_nrmse=worst,
                yields=yields.tolist(),
            )
        )
        if family not in best or worst < best[family][0]:
            best[family] = worst, pred, yields, lag_h, removal_h
        return worst

    for family in ("additive", "mixture_response"):
        for lag_h in (1 / 12, 1 / 6, 1 / 3, 0.5, 2 / 3, 1, 1.5, 2, 3, 4):
            for removal_h in (
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                0.60,
                0.8,
                1.0,
                1.5,
                2,
                3,
            ):
                evaluate((lag_h, removal_h), family, "grid")
        top = sorted(
            (r for r in designs if r["family"] == family),
            key=lambda r: r["worst_nrmse"],
        )[:3]
        for row in top:
            minimize(
                evaluate,
                [row["lag_h"], row["removal_h"]],
                args=(family, "refinement"),
                method="Nelder-Mead",
                bounds=[(1 / 12, 4), (0.05, 3)],
                options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 200},
            )
    family = "parent_lags"
    limits = np.array([(1 / 12, 4)] * 3 + [(0.05, 3)])
    anchor = best["mixture_response"]
    evaluate([anchor[3]] * 3 + [anchor[4]], family, "nested")
    for seed in (317, 883):
        unit = qmc.Sobol(4, scramble=True, seed=seed).random_base2(7)
        points = np.exp(
            np.log(limits[:, 0]) + unit * np.log(limits[:, 1] / limits[:, 0])
        )
        trials = [(evaluate(p, family, "sobol"), p) for p in points]
        for _, point in sorted(trials, key=lambda item: item[0])[:2]:
            minimize(
                evaluate,
                point,
                args=(family, "refinement"),
                method="Nelder-Mead",
                bounds=limits,
                options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 250},
            )
    for family, (_, predictions, yields, lag, removal) in best.items():
        for (r, _, t, y, _), p, s in zip(
            runs, predictions, scales, strict=True
        ):
            scores.append(
                dict(
                    family=family,
                    run=r,
                    n=len(y),
                    duration_h=t[-1] / 3600,
                    scale_ug_m3=s,
                    rmse=np.mean((p - y) ** 2) ** 0.5,
                    nrmse=np.mean((p - y) ** 2) ** 0.5 / s,
                    obs_peak=y.max(),
                    model_peak=p.max(),
                    lag_h=lag,
                    removal_h=removal,
                    yields=yields.tolist(),
                )
            )
            curves.extend(
                dict(family=family, run=r, time_s=tt, measured=yy, modeled=pp)
                for tt, yy, pp in zip(t, y, p, strict=True)
            )
    for name, rows in (
        ("campaign_scores", scores),
        ("campaign_grid", designs),
        ("campaign_curves", curves),
    ):
        pd.DataFrame(rows).to_csv(
            HERE / "results" / (name + ".csv"), index=False
        )
