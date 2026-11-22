# %%
"""Re-fit campaign yields and bounded run doses at the selected kinetics."""

import json

import numpy as np
import pandas as pd
from campaign import HERE, design_matrices, inputs
from scipy.optimize import linprog, minimize
from scipy.stats import qmc


def dose_fit(matrices, targets, scales):
    """Bound the joint RMSE using linear cuts of second-order cones."""
    n = len(targets)
    k = matrices[0].shape[1]
    factored = [
        np.linalg.qr(np.column_stack((x, -y)) / (s * np.sqrt(len(y))), mode="r")
        for x, y, s in zip(matrices, targets, scales, strict=True)
    ]
    # v = 1/dose. A bound t on each normalized RMSE is equivalent to
    # ||R_i [yields, v_i]|| <= t*v_i, a convex constraint for fixed t.
    bounds = [(0, 2)] * k + [(1 / 1.15, 1 / 0.85)] * n
    cuts = []
    owners = []
    best = np.r_[np.zeros(k), np.ones(n)]
    lower = 0.0
    upper = max(np.linalg.norm(r[:, -1]) for r in factored)
    for _ in range(22):
        threshold = (lower + upper) / 2
        for _ in range(500):
            a = np.array(cuts).reshape(-1, k + n)
            a[np.arange(len(owners)), k + np.array(owners, dtype=int)] -= (
                threshold
            )
            lp = linprog(
                np.zeros(k + n),
                A_ub=a if len(a) else None,
                b_ub=np.zeros(len(a)) if len(a) else None,
                bounds=bounds,
            )
            if lp.status == 2:
                lower = threshold
                break
            if not lp.success:
                raise ArithmeticError(lp.message)
            # Keep the constructive witness inside the exact dose bounds.
            x = np.clip(lp.x, *np.asarray(bounds).T)
            violations = 0
            errors = []
            for i, r in enumerate(factored):
                v = x[k + i]
                error = r @ np.r_[x[:k], v]
                norm = np.linalg.norm(error)
                errors.append(norm / v)
                if norm / v <= threshold + 2e-6:
                    continue
                row = np.zeros(k + n)
                grad = (error / norm) @ r
                row[:k], row[k + i] = grad[:-1], grad[-1]
                cuts.append(row)
                owners.append(i)
                violations += 1
            if not violations:
                upper = max(errors)
                best = x
                break
        else:
            raise RuntimeError("Dose and yield feasibility failed to converge")
        if upper - lower <= 5e-6:
            break
    return best[:k], 1 / best[k:], lower, upper


if __name__ == "__main__":
    runs = inputs()
    base = pd.read_csv(HERE / "results/campaign_scores.csv")
    targets = [r[3] for r in runs]
    scales = [max(1.0, y.max()) for y in targets]
    out, curves = [], []
    for family, group in base.groupby("family"):
        row = group.iloc[0]
        w = row.removal_h
        lag = json.loads(row.lag_h)
        mats = design_matrices(runs, lag, w, family)
        solution = dose_fit(mats, targets, scales)
        best = dict(fixed=(solution, mats, lag, w))
        bounds = [(1 / 12, 4)] * (3 if isinstance(lag, list) else 1) + [
            (0.05, 3)
        ]
        history = []

        def evaluate(pair, family=family, best=best, history=history):
            """Profile the convex joint dose and yield fit at chosen rates."""
            lags = (
                list(map(float, pair[:-1]))
                if len(pair) == 4
                else float(pair[0])
            )
            matrices = design_matrices(runs, lags, pair[-1], family)
            answer = dose_fit(matrices, targets, scales)
            if "joint" not in best or answer[-1] < best["joint"][0][-1]:
                best["joint"] = answer, matrices, lags, pair[-1]
            history.append((answer[-1], np.array(pair)))
            return answer[-1]

        evaluate(np.r_[lag, w])
        limits = np.asarray(bounds)
        for v in qmc.Sobol(len(bounds), scramble=True, seed=8103).random_base2(
            6
        ):
            evaluate(
                np.exp(
                    np.log(limits[:, 0])
                    + v * np.log(limits[:, 1] / limits[:, 0])
                )
            )
        for _, point in sorted(history, key=lambda a: a[0])[:2]:
            minimize(
                evaluate,
                point,
                method="Nelder-Mead",
                bounds=bounds,
                options={"maxiter": 150, "xatol": 1e-4, "fatol": 1e-6},
            )
        for stage, ((yields, doses, low, high), mats, lag, w) in best.items():
            for r, x, y, s, d in zip(
                runs, mats, targets, scales, doses, strict=True
            ):
                pred = x @ yields * d
                if family == "parent_lags" and stage == "joint":
                    curves.extend(
                        dict(run=r[0], time_s=t, modeled=m)
                        for t, m in zip(r[2], pred, strict=True)
                    )
                out.append(
                    dict(
                        family=family,
                        stage=stage,
                        lag_h=json.dumps(lag),
                        removal_h=w,
                        run=r[0],
                        dose_multiplier=d,
                        lower_worst_nrmse=low,
                        upper_worst_nrmse=high,
                        nrmse=np.mean((pred - y) ** 2) ** 0.5 / s,
                        model_peak=pred.max(),
                        obs_peak=y.max(),
                        yields=json.dumps(yields.tolist()),
                    )
                )
    pd.DataFrame(out).to_csv(
        HERE / "results/campaign_dose_scores.csv", index=False
    )

    pd.DataFrame(curves).to_csv(
        HERE / "results/campaign_dose_curves.csv", index=False
    )
