# %%
"""Search globally for shared mechanisms with equal priority per run."""

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy
import warp as wp
from engine import ORDER, RESULT, Engine, observations
from scipy.optimize import differential_evolution
from scipy.stats import qmc

# Parameters with ranges over orders of magnitude use log10 coordinates.
NAMES = (
    "branch",
    "log_Cstar",
    "log_accommodation",
    "log_lag_min",
    "particle_wall_h",
    "vapor_wall_h",
    "direct_low_fraction",
    "log_volatility_ratio",
    "aging_h",
    "dose1",
    "dose4",
    "dose5",
    "dose6",
    "dose2",
    "wall_share",
    "competition_h_at_10",
    "branch_NOx_slope",
)
PHYSICAL_NAMES = (
    "branch_at_30ppb",
    "Cstar_S_ug_m3",
    "accommodation",
    "lag_min",
    "particle_wall_h",
    "vapor_wall_h",
    "primary_LV_share",
    "Cstar_S_over_L",
    "aging_h",
    "dose1",
    "dose4",
    "dose5",
    "dose6",
    "dose2",
    "wall_share",
    "pair_h_at_10ug_m3",
    "NOx_slope",
)
BOUNDS = np.array(
    [
        (0.05, 0.85),
        (-0.5, 2.5),
        (-2, -0.25),
        (0.3, 1.85),
        (0.04, 0.65),
        (0.0, 5.0),
        (0.0, 1.0),
        (0.0, 2.5),
        (0.0, 4.0),
        *((0.85, 1.15),) * 5,
        (0.0, 1.0),
        (0.0, 12.0),
        (-3.0, 3.0),
    ]
)
DEFAULT = np.array(
    [
        0.25,
        0.0,
        -1.5,
        0.7,
        0.42,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
    ]
)
FAMILIES = (
    "single",
    "expanded",
    "single_doses",
    "expanded_doses",
    "expanded_counts",
    "five_doses",
    "competition",
    "competition_doses",
    "competition_counts",
    "single_nox",
    "expanded_nox",
    "competition_nox_counts",
    "single_nox_five",
)
SEEDS = (1941, 3527, 7819)


def definition(family):
    """Return active coordinates, number closure and fitted runs."""
    active = list(range(6))
    if not family.startswith("single"):
        active.extend([6, 7, 8])
    if "doses" in family:
        active.extend(range(9, 14 if family == "five_doses" else 13))
    if "nox" in family:
        active.append(16)
    if family.startswith("competition"):
        active.append(15)
    if family.endswith("counts"):
        active.remove(4)
        active.append(14)
    return (
        active,
        family.endswith("counts"),
        5 if family in ("five_doses", "single_nox_five") else 4,
    )


def physical(coords, active):
    """Expand search coordinates to the physical parameters."""
    x = np.tile(DEFAULT, (len(coords), 1))
    x[:, active] = coords
    for j in (1, 2, 3, 7):
        x[:, j] = 10 ** x[:, j]
    return x


class Score:
    """Score trajectories at the original AMS sampling times."""

    def __init__(self):
        """Cache interpolation weights at the original AMS times."""
        self.data = []
        for run in ORDER:
            t, y = observations(run)
            y = y[(t >= 0) & (t <= 18000)]
            t = t[(t >= 0) & (t <= 18000)]
            ix = np.minimum((t / 360).astype(int), 49)
            w = t / 360 - ix
            self.data.append((ix, w, y))

    def errors(self, h):
        """Compute peak-normalized trajectory RMSE for each run."""
        out = []
        for i, (idx, w, y) in enumerate(self.data):
            p = (1 - w) * h[:, i, idx] + w * h[:, i, idx + 1]
            out.append(np.sqrt(np.mean((p - y) ** 2, axis=1)) / y.max())
        return np.stack(out, axis=1)

    def objective(self, h, n):
        """Return the worst normalized error across the fitted runs."""
        return self.errors(h)[:, :n].max(axis=1)


def fit(family, coarse, fine, score):
    """Use independent global searches and bounded local perturbations."""
    active, counts, n = definition(family)
    bounds = BOUNDS[active]
    low, high = bounds.T
    records = []
    archive = []
    start = time.monotonic()

    def eval_many(x, engine):
        sols = physical(np.atleast_2d(x), active)
        return score.objective(engine(sols, counts), n)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        points = qmc.Sobol(len(active), scramble=True, seed=seed).random_base2(
            11
        )
        init = low + points * (high - low)
        init[0] = DEFAULT[active]
        ancestry = {
            "single_nox": ("single",),
            "single_nox_five": ("single_nox",),
            "expanded_nox": ("single_nox", "expanded"),
            "competition_nox_counts": ("competition_counts", "expanded_nox"),
            "expanded": ("single",),
            "single_doses": ("single",),
            "expanded_doses": ("expanded", "single_doses"),
            "five_doses": ("expanded_doses",),
            "competition": ("expanded",),
            "competition_doses": ("expanded_doses", "competition"),
            "competition_counts": ("expanded_counts",),
        }
        parents = []
        for parent in ancestry.get(family, ()):
            prior = json.loads((RESULT / (parent + ".json")).read_text())
            parents.append(np.array(prior["coordinates"])[active])
        init[1 : 1 + len(parents)] = np.asarray(parents).reshape(
            -1, len(active)
        )
        rank = eval_many(init, coarse)
        # Retain competing starts and spread the remainder across the domain.
        elite = init[np.argsort(rank)[:32]]
        pop = np.vstack([elite, init[rng.choice(len(init), 96, replace=False)]])

        def vectorized(x):
            return eval_many(np.asarray(x).T, coarse)

        answer = differential_evolution(
            vectorized,
            bounds,
            init=pop,
            rng=rng,
            maxiter=85,
            polish=False,
            updating="deferred",
            vectorized=True,
            mutation=(0.4, 0.95),
            recombination=0.84,
            tol=1e-5,
        )
        candidates = np.vstack([answer.population, init[: 1 + len(parents)]])
        energies = eval_many(candidates, fine)
        best = candidates[energies.argmin()].copy()
        value = float(energies.min())
        # Population refinement of a nonsmooth minimax score within the bounds.
        for scale in np.geomspace(0.025, 0.00015, 32):
            trial = np.clip(
                best + rng.normal(size=(128, len(best))) * (high - low) * scale,
                low,
                high,
            )
            loss = eval_many(trial, fine)
            idx = loss.argmin()
            if loss[idx] < value:
                best = trial[idx]
                value = float(loss[idx])
        coords = DEFAULT.copy()
        coords[active] = best
        record = dict(
            seed=seed,
            objective=value,
            coordinates=coords.tolist(),
            physical=physical(best[None, :], active)[0].tolist(),
            generations=int(answer.nit),
            sampled=2048
            + 128 * (int(answer.nit) + 1)
            + len(candidates)
            + 32 * 128,
        )
        records.append(record)
        for j in np.argsort(energies)[:12]:
            archive.append(
                dict(
                    seed=seed,
                    objective=float(energies[j]),
                    **dict(
                        zip(
                            PHYSICAL_NAMES,
                            physical(candidates[j : j + 1], active)[0],
                            strict=True,
                        )
                    ),
                )
            )
        print(
            family,
            seed,
            value,
            round(time.monotonic() - start, 1),
            "s",
            flush=True,
        )
    top = min(records, key=lambda row: row["objective"])
    payload = dict(
        **top,
        family=family,
        coordinates_names=NAMES,
        active=[NAMES[j] for j in active],
        restarts=records,
        seconds=time.monotonic() - start,
        dt_s=fine.dt,
        coarse_dt_s=coarse.dt,
        physical_names=PHYSICAL_NAMES,
        method="multi-fidelity differential evolution",
        device=wp.get_device("cuda:0").name,
        scipy=scipy.__version__,
        warp=wp.__version__,
    )
    (RESULT / (family + ".json")).write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("families", nargs="*", default=list(FAMILIES))
    args = parser.parse_args()
    device = wp.get_device("cuda:0")
    if not device.is_cuda:
        raise RuntimeError("CUDA required")
    coarse = Engine(dt=18)
    fine = Engine(dt=3)
    score = Score()
    for f in args.families:
        if f not in FAMILIES:
            raise ValueError(f)
        pd.DataFrame(fit(f, coarse, fine, score)).to_csv(
            RESULT / (f + "_population.csv"), index=False
        )
