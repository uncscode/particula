# %%
"""Search for joint trajectory and peak agreement in every run."""

import json

import numpy as np
import pandas as pd
from engine import RESULT, Engine
from optimize import BOUNDS, DEFAULT, Score, definition, physical
from scipy.optimize import differential_evolution
from scipy.stats import qmc

FAMILIES = (
    "single_nox",
    "competition_nox_counts",
    "expanded_nox",
    "single",
    "expanded",
    "expanded_counts",
    "single_doses",
    "five_doses",
)


def gate_score(engine, x, family, scorer):
    """Return the largest error relative to its tolerance."""
    active, counts, n = definition(family)
    h = engine(physical(np.atleast_2d(x), active), counts)
    peaks = np.array([y.max() for _, _, y in scorer.data])
    traj = scorer.errors(h)[:, :n].max(axis=1) / 0.10
    peak = (abs(h.max(axis=2) / peaks - 1)[:, :n]).max(axis=1) / 0.15
    return np.maximum(traj, peak)


if __name__ == "__main__":
    coarse, fine = Engine(dt=18), Engine(dt=3)
    scorer = Score()
    rows = []
    for f in FAMILIES:
        active, counts, n = definition(f)
        bounds = BOUNDS[active]
        lo, hi = bounds.T
        base = DEFAULT[active]
        for seed in (1410, 6004):
            rng = np.random.default_rng(seed)
            initial = lo + qmc.Sobol(
                len(active), scramble=True, seed=seed
            ).random_base2(10) * (hi - lo)
            initial[0] = base
            v = gate_score(coarse, initial, f, scorer)
            pop = np.vstack(
                [
                    initial[np.argsort(v)[:24]],
                    initial[rng.choice(len(v), 72, replace=False)],
                ]
            )

            def objective(x, family=f):
                """Score the population supplied by differential evolution."""
                return gate_score(coarse, np.asarray(x).T, family, scorer)

            de = differential_evolution(
                objective,
                bounds,
                rng=rng,
                init=pop,
                maxiter=60,
                tol=1e-5,
                polish=False,
                vectorized=True,
                updating="deferred",
            )
            pool = np.vstack([de.population, base])
            v = gate_score(fine, pool, f, scorer)
            x = pool[v.argmin()]
            val = v.min()
            for scale in np.geomspace(0.035, 0.0001, 36):
                trials = np.clip(
                    x + rng.normal(size=(128, len(x))) * (hi - lo) * scale,
                    lo,
                    hi,
                )
                vals = gate_score(fine, trials, f, scorer)
                k = np.argmin(vals)
                if vals[k] < val:
                    x = trials[k]
                    val = vals[k]
            p = physical(x[None, :], active)[0]
            trace = fine(p, counts)[0]
            peak = np.array([y.max() for _, _, y in scorer.data])
            rows.append(
                dict(
                    family=f,
                    seed=seed,
                    gate_ratio=val,
                    passes=bool(val <= 1),
                    worst_nrmse=scorer.errors(trace[None, :, :])[0, :n].max(),
                    max_abs_peak_error=abs(trace.max(axis=1) / peak - 1)[
                        :n
                    ].max(),
                    physical=json.dumps(p.tolist()),
                )
            )
            print(f, seed, val, flush=True)
        pd.DataFrame(rows).to_csv(RESULT / "joint_gate_search.csv", index=False)
