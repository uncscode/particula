# %%
"""Compare additional product pathways across the fifteen Manchester runs."""

import json
import platform

import numpy as np
import pandas as pd
import particula
import scipy
import warp as wp
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from uncertainty_profile import NAMES, anchors
from wall_campaign import HERE, Campaign

OUT = HERE / "results/pathways"
MODELS = {
    "reference": (0, []),
    "prompt": (1, [10]),
    "auto_competition": (2, [10, 11, 12]),
    "reactive_uptake": (3, [13, 14, 15]),
    "combined": (4, [10, 11, 12, 13, 14, 15]),
}
EXTRA = [
    "alpha_branch_g_g",
    "auto_time_s",
    "competition_per_ppb_min",
    "uptake_precursor_g_g",
    "effective_gamma",
    "vapor_loss_h_inv",
]
LIMITS = np.array(
    [(1 / 12, 4)] * 3
    + [(0, 2)] * 6
    + [
        (2.818382931264455, 16.551404797047777),
        (0, 0.25),
        (0.1, 300.0),
        (0, 10.0),
        (0, 0.25),
        (1e-5, 0.03),
        (0.05, 5.0),
    ]
)
LOG = [0, 1, 2, 9, 11, 14, 15]
SEEDS = (2371, 8191)
DOSES = np.linspace(0.85, 1.15, 61)
DEFAULT = [0.0, 10.0, 1.0, 0.0, 0.001, 1.0]


def transformed(p):
    """Use log coordinates for rate scales and delays."""
    x = np.atleast_2d(p).copy()
    x[:, LOG] = np.log10(x[:, LOG])
    return x


def physical(x):
    """Recover the physical parameters used in the CUDA integration."""
    p = np.atleast_2d(x).copy()
    p[:, LOG] = 10.0 ** p[:, LOG]
    return p


def fit(c, mode, active, objective, seeds_in, seed, limits=LIMITS):
    """Search globally and refine around near-optimal and nested candidates."""
    rng = np.random.default_rng(seed)
    low, high = transformed(limits.T)[:, active]
    prior = transformed(seeds_in)
    template = prior[0]

    def expand(x):
        full = np.tile(template, (len(np.atleast_2d(x)), 1))
        full[:, active] = np.atleast_2d(x)
        return physical(full)

    def score(x, doses, sub=1):
        return c.evaluate(
            expand(x), True, objective, doses, sub=sub, pathway=mode
        )[0]

    coarse = DOSES[::30]
    fine = DOSES[::6]
    d = len(active)
    sample = low + qmc.Sobol(d, seed=seed, scramble=True).random_base2(8) * (
        high - low
    )
    # Half the population respects the zero mass observed for pure isoprene.
    sample[::2, active.index(5)] = 0.0
    start = prior[:, active]
    near = np.clip(
        start[rng.integers(len(start), size=128)]
        + rng.normal(size=(128, d)) * 0.045 * (high - low),
        low,
        high,
    )
    near[::2, active.index(5)] = 0.0
    sample = np.vstack((start, near, sample))
    ranking = np.argsort(score(sample, coarse))
    population = np.vstack(
        (
            sample[ranking[:24]],
            sample[rng.choice(len(sample), 36, replace=False)],
        )
    )
    ans = differential_evolution(
        lambda x: score(x.T, coarse),
        list(zip(low, high, strict=True)),
        init=population,
        maxiter=42,
        rng=rng,
        vectorized=True,
        updating="deferred",
        polish=False,
        tol=1e-6,
    )
    pool = np.vstack((start, ans.population, sample[ranking[:24]]))
    scores = score(pool, fine)
    best = pool[scores.argmin()]
    for scale in np.geomspace(0.055, 0.00012, 30):
        random = rng.normal(size=(48, d))
        # Include one-coordinate moves near nonsmooth minimax solutions.
        steps = np.vstack((np.zeros(d), random, np.eye(d), -np.eye(d)))
        trials = np.clip(best + steps * scale * (high - low), low, high)
        trials[2:24:2, active.index(5)] = 0.0
        best = trials[score(trials, fine).argmin()]
    candidates = expand(np.vstack((best, start)))
    scores = c.evaluate(
        candidates, True, objective, DOSES, sub=2, pathway=mode
    )[0]
    return candidates[scores.argmin()], scores.min()


def check(c, base):
    """Check the nested limit, archived scores, and active product budgets."""
    from wall_campaign import OUT as WALL

    ref = c.evaluate(base, True, "joint", DOSES, sub=2, outcomes=True)
    archived = pd.read_csv(WALL / "campaign_scores.csv").query(
        "model=='eddy_brownian' and objective=='joint'"
    )
    np.testing.assert_allclose(
        ref[1][0], archived[["mass_nrmse", "count_nrmse"]], atol=2e-12
    )
    for mode, _ in MODELS.values():
        p = np.r_[base, DEFAULT]
        nested = c.evaluate(
            p, True, "joint", DOSES, sub=2, outcomes=True, pathway=mode
        )
        for a, b in zip(ref, nested, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    p[[10, 13]] = 0.07
    out = c.evaluate(
        p, True, "joint", np.array([1.0]), sub=2, pathway=4, traces=True
    )[3]
    if out.min() < -1e-8 or min(out[0, :, 3].max(), out[0, :, 4].max()) <= 0:
        raise ArithmeticError("Active pathway check failed")


def main():
    """Run the nested fits at a common wall setting and with bounded mixing."""
    OUT.mkdir(exist_ok=True)
    c = Campaign()
    base, _, joint, dark, mass = anchors()
    check(c, base)
    env = dict(
        python=platform.python_version(),
        particula=particula.__version__,
        scipy=scipy.__version__,
        numpy=np.__version__,
        warp=wp.__version__,
        device=wp.get_device("cuda:0").name,
        seeds=SEEDS,
        bounds=dict(zip(NAMES + tuple(EXTRA), LIMITS.tolist(), strict=True)),
    )
    (OUT / "design.json").write_text(json.dumps(env, indent=2) + "\n")
    history = []
    for objective, free in [("mass", False), ("joint", False), ("joint", True)]:
        for model, (mode, extra) in MODELS.items():
            active = list(range(9)) + ([9] if free else []) + extra
            seeds_in = [
                np.r_[r["physical"], DEFAULT] for r in joint + mass + dark
            ]
            # Transfer earlier fits while preserving the nesting of each model.
            for r in history:
                p = np.array(r["physical"])
                for i in set(range(10, 16)) - set(extra):
                    p[i] = DEFAULT[i - 10]
                seeds_in.append(p)
            for p in seeds_in:
                p[9] = np.clip(p[9], *LIMITS[9]) if free else base[9]
            seeds_in = np.unique(seeds_in, axis=0)
            for seed in SEEDS:
                p, s = fit(c, mode, active, objective, seeds_in, seed)
                history.append(
                    dict(
                        model=model,
                        objective=objective,
                        wall_free=free,
                        seed=seed,
                        score=float(s),
                        physical=p.tolist(),
                    )
                )
                print(model, objective, free, seed, round(s, 5), flush=True)
            (OUT / "restarts.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
