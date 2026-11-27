"""Generate node-resolved ODE targets and randomized vapor forcing."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from sparse_vapor_reference import ExplicitReservoir
from stage2_population import fixture


def trajectory(box, tol=1e-9, active=None):
    """Resolve particle-level loadings and gas with an independent ODE."""
    s = box.spec.species
    model = ExplicitReservoir(
        box.mass, np.ones(len(box.mass)), box.volume, box.spec
    )
    q = (box.mass[:, :s] / model.local.dry[:, None]).ravel()

    def rhs(t, y):
        result = model.rates(t, y)
        if active is not None:
            result[:-s].reshape(-1, s)[:, ~active] = 0.0
            result[-s:][~active] = 0.0
        return result

    gas = box.gas.copy()
    qs, gs = [], []
    starts = np.r_[0, np.flatnonzero(np.diff(box.t) == 0) + 1]
    for leg, (i, j) in enumerate(
        zip(starts, np.r_[starts[1:], len(box.t)], strict=True)
    ):
        gas = np.where(box.masks[leg], box.levels[leg], gas)
        t = box.t[i:j] - box.t[i]
        sol = solve_ivp(
            rhs,
            (0.0, t[-1]),
            np.r_[q, gas / model.scale],
            method="LSODA",
            t_eval=t,
            rtol=tol,
            atol=tol * 1e-3,
        )
        if not sol.success or sol.y.min() < -tol * 0.03:
            raise ArithmeticError(sol.message)
        q = sol.y[:-s, -1]
        gas = sol.y[-s:, -1] * model.scale
        qs.append(sol.y[:-s].T.reshape(len(t), -1, s))
        gs.append(sol.y[-s:].T * model.scale)
    return np.concatenate(qs), np.concatenate(gs)


def generate(args):
    """Create one member of a matched pair with an optional isolated process."""
    family, mix, density, process, frames, slots, phase, active = args
    rng = np.random.default_rng([77381, family])
    duration = (
        float(rng.choice([0.002, 0.005, 0.01, 0.02, 0.04, 0.08]))
        if phase == "train"
        else 0.04
    )
    box = fixture(
        family, mix, slots, density, process, frames=frames, duration=duration
    )
    lane = (
        np.arange(box.spec.species) == (0 if active == "water" else 1)
        if active != "all"
        else None
    )
    q, g = trajectory(box, active=lane)
    return dict(
        m=box.mass,
        number=np.full(slots, 1 / box.volume),
        levels=box.levels,
        mask=box.masks,
        q=q,
        g=g,
        t=box.t,
        ids=family,
        mix=mix,
        density=density,
    )


def run(out, first, families, phase, process, frames, slots, workers, active):
    """Save native trajectories, forcing and population weights."""
    out.parent.mkdir(exist_ok=True, parents=True)
    tasks = (
        (f, m, d, process, frames, slots, phase, active)
        for f in range(first, first + families)
        for m in (0, 1)
        for d in (1, 8)
    )
    with ProcessPoolExecutor(workers) as pool:
        data = list(pool.map(generate, tasks, chunksize=2))
    np.savez_compressed(
        out, **{k: np.stack([x[k] for x in data]) for k in data[0]}
    )
    print(out, len(data), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("first", type=int)
    p.add_argument("families", type=int)
    p.add_argument("--phase", default="train")
    p.add_argument("--process", default="both")
    p.add_argument(
        "--active", choices=["all", "water", "organic"], default="all"
    )
    for k, v in (("frames", 8), ("slots", 16), ("workers", 3)):
        p.add_argument("--" + k, type=int, default=v)
    run(**vars(p.parse_args()))
