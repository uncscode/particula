# %%
"""Fit chamber wall mixing to organic mass and particle counts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from engine import CONDITIONS, ORDER, RESULT, TIMES, Engine
from optimize import BOUNDS, Score
from scipy.optimize import differential_evolution
from scipy.stats import qmc

OUT = RESULT / "wall"
ACTIVE = [0, 1, 2, 3, 5, 16]
SEEDS = (2187, 6741, 8105)
MODELS = ("constant", "eddy", "constant_brownian", "eddy_brownian")
RUNS = ORDER[:4]


class WallObjective:
    """Match five-hour mass and count records at their own timestamps."""

    def __init__(self):
        """Reuse the initial counts and instruments from the chemistry study."""
        self.mass = Score()
        context = pd.read_csv(
            Path(__file__).resolve().parents[1] / "data/context.csv"
        )
        self.counts = []
        for r in RUNS:
            channel = "Total_number" if r == 5 else "total_NumberConc"
            data = context[(context.run == r) & (context.channel == channel)]
            data = data[data.time_s.between(0, 18000)].sort_values("time_s")
            t = data.time_s.to_numpy()
            h = t / 360
            idx = np.minimum(h.astype(int), len(TIMES) - 2)
            observed = data.value.to_numpy() / CONDITIONS[r][2]
            self.counts.append((idx, h - idx, observed, t))

    def errors(self, mass, number):
        """Return one independent mass and count trajectory error per run."""
        errors = []
        for i, (idx, w, obs, _) in enumerate(self.counts):
            pred = (1 - w) * number[:, i, idx] + w * number[:, i, idx + 1]
            errors.append(np.sqrt(np.mean((pred - obs) ** 2, axis=1)))
        return self.mass.errors(mass)[:, :4], np.stack(errors, axis=1)


def simulate(engine, coords, model, original):
    """Evaluate one or more vectors with the indicated particle loss model."""
    x = np.atleast_2d(coords)
    p = np.tile(original, (len(x), 1))
    p[:, ACTIVE] = x[:, :6]
    p[:, [1, 2, 3]] = 10 ** p[:, [1, 2, 3]]
    wall_value = 10 ** x[:, -1]
    eddy = wall_value if model.startswith("eddy") else None
    if eddy is None:
        p[:, 4] = wall_value
    return engine(
        p, eddy_s=eddy, brownian=model.endswith("brownian"), return_numbers=True
    )


def objective_values(mass_errors, count_errors, objective):
    """Give each instrument and experiment equal priority in joint mode."""
    if objective == "mass":
        return mass_errors.max(1)
    return np.maximum(mass_errors.max(1), count_errors.max(1))


def fit(model, objective, original, engines, score, start, fixed_eddy=None):
    """Apply global evolutionary search and rescore/refine on a finer grid."""
    bounds = np.vstack(
        [
            BOUNDS[ACTIVE],
            (-3, 3)
            if model.startswith("eddy")
            else tuple(np.log10([0.04, 0.65])),
        ]
    )
    if fixed_eddy is not None:
        bounds[-1, :] = np.log10(fixed_eddy)
        start = start.copy()
        start[-1] = np.log10(fixed_eddy)
    lo, hi = bounds.T

    def loss(x, engine):
        return objective_values(
            *score.errors(*simulate(engine, x, model, original)), objective
        )

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        init = lo + qmc.Sobol(7, scramble=True, seed=seed).random_base2(10) * (
            hi - lo
        )
        init[0] = start
        ranks = np.argsort(loss(init, engines[0]))
        pop = np.vstack(
            (init[ranks[:24]], init[rng.choice(len(init), 72, replace=False)])
        )
        result = differential_evolution(
            lambda x: loss(x.T, engines[0]),
            bounds,
            init=pop,
            vectorized=True,
            updating="deferred",
            rng=rng,
            maxiter=65,
            tol=2e-6,
            polish=False,
        )
        pool = np.vstack((result.population, start))
        values = loss(pool, engines[1])
        best = pool[values.argmin()]
        value = values.min()
        for scale in np.geomspace(0.025, 0.0001, 26):
            trials = np.clip(
                best + rng.normal(size=(96, 7)) * scale * (hi - lo), lo, hi
            )
            vals = loss(trials, engines[1])
            if vals.min() < value:
                best = trials[vals.argmin()]
                value = vals.min()
        rows.append(
            dict(
                model=model,
                objective=objective,
                seed=seed,
                score=float(value),
                coords=best.tolist(),
                wall_parameter=10 ** best[-1],
            )
        )
        print(
            model,
            objective,
            seed,
            round(value, 5),
            round(10 ** best[-1], 4),
            flush=True,
        )
    return rows


def main():
    """Fit matched baselines and save scores and sensitivities."""
    OUT.mkdir(exist_ok=True)
    original = np.array(
        json.loads((RESULT / "single_nox.json").read_text())["physical"]
    )
    coords = original[ACTIVE].copy()
    coords[1:4] = np.log10(coords[1:4])
    engines = [Engine(18), Engine(3)]
    score = WallObjective()
    records = []
    metrics = []
    traces = []
    profiles = []
    for model in MODELS:
        start = np.r_[
            coords, np.log10(5 if model.startswith("eddy") else original[4])
        ]
        for obj in (
            ("joint", "mass") if model == "eddy_brownian" else ("joint",)
        ):
            fits = fit(model, obj, original, engines, score, start)
            records.extend(fits)
            best = min(fits, key=lambda x: x["score"])
            x = np.array(best["coords"])
            mass, number = simulate(engines[1], x, model, original)
            refined = simulate(Engine(2), x, model, original)
            me, ne = score.errors(mass, number)
            for i, r in enumerate(RUNS):
                metrics.append(
                    dict(
                        model=model,
                        objective=obj,
                        run=r,
                        wall_parameter=best["wall_parameter"],
                        mass_nrmse=me[0, i],
                        count_nrmse=ne[0, i],
                        end_number_fraction=number[0, i, -1],
                        end_observed_fraction=score.counts[i][2][-1],
                        last_observed_s=score.counts[i][-1][-1],
                        refine_mass_ug_m3=np.max(
                            abs(mass[0, i] - refined[0][0, i])
                        ),
                        refine_count_fraction=np.max(
                            abs(number[0, i] - refined[1][0, i])
                        ),
                    )
                )
                traces.extend(
                    dict(
                        model=model,
                        objective=obj,
                        run=r,
                        time_s=t,
                        soa_ug_m3=m,
                        number_fraction=n,
                    )
                    for t, m, n in zip(
                        TIMES, mass[0, i], number[0, i], strict=True
                    )
                )
            if model.startswith("eddy") and obj == "joint":
                grid = np.tile(x, (121, 1))
                grid[:, -1] = np.linspace(-3, 3, len(grid))
                m, n = score.errors(
                    *simulate(engines[1], grid, model, original)
                )
                for i, r in enumerate(RUNS):
                    profiles.extend(
                        dict(
                            model=model,
                            run=r,
                            eddy_s=10 ** g[-1],
                            mass_nrmse=mm,
                            count_nrmse=nn,
                        )
                        for g, mm, nn in zip(
                            grid, m[:, i], n[:, i], strict=True
                        )
                    )
            pd.DataFrame(metrics).to_csv(OUT / "scores.csv", index=False)
            (OUT / "restarts.json").write_text(
                json.dumps(records, indent=2) + "\n"
            )
            pd.DataFrame(traces).to_csv(OUT / "trajectories.csv", index=False)
            pd.DataFrame(profiles).to_csv(
                OUT / "conditional_profiles.csv", index=False
            )
            start = x


if __name__ == "__main__":
    main()
