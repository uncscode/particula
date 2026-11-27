"""Generate physical targets on states reached by learned advances."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sparse_vapor_reference import ExplicitReservoir
from stage2_population import fixture
from stage2_transfer import restore, step_box
from stage2_transfer_data import trajectory


def physical(box, dt):
    """Advance the current accepted state with an independent vapor ODE."""
    n = len(box.mass)
    s = box.spec.species
    ode = ExplicitReservoir(box.mass, np.ones(n), box.volume, box.spec)
    initial = np.r_[
        (box.mass[:, :s] / ode.local.dry[:, None]).ravel(), box.gas / ode.scale
    ]
    sol = solve_ivp(
        ode.rates,
        (0.0, dt),
        initial,
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
        t_eval=[dt],
    )
    if not sol.success:
        raise ArithmeticError(sol.message)
    y = sol.y[:, -1]
    box.mass[:, :s] = y[:-s].reshape(n, s) * ode.local.dry[:, None]
    box.gas = y[-s:] * ode.scale


_MODELS = {}


@torch.inference_mode()
def sample(job):
    """Draw a new training anchor from an actual mixed-policy rollout."""
    family, mix, density, seed, model, substeps = job
    rng = np.random.default_rng([family, mix, density, seed])
    torch.set_num_threads(1)
    duration = float(rng.choice([0.002, 0.005, 0.01, 0.02, 0.04, 0.08]))
    box = fixture(family, mix, 16, density, duration=duration)
    if model not in _MODELS:
        _MODELS[model] = restore(box.spec, model)
    net = _MODELS[model]
    mode = int(rng.integers(4))
    anchor = int(rng.integers(1, len(box.t)))
    # Some anchors lie just after a forcing event and others late on a plateau.
    learned = 0
    phys = 0
    leg = 0
    age = 0
    h = None
    last = "initial"
    for dt in np.diff(box.t[: anchor + 1]):
        if dt == 0:
            leg += 1
            age = 0
            box.gas = np.where(box.masks[leg], box.levels[leg], box.gas)
            continue
        use = (
            (mode == 1 and age == 0)
            or (mode == 2 and (age + 1) % 4 == 0)
            or (mode == 3 and rng.random() < 0.25)
        )
        if use:
            physical(box, dt)
            h = None
            phys += 1
            last = "ODE"
        else:
            h = step_box(box, net, dt, substeps, h)
            learned += 1
            last = "learned"
        age += 1
    # The shifted student state is now the initial condition for an
    # independently
    # integrated continuation. Its actual gas is preserved at the new origin.
    box.levels = np.roll(box.levels, -leg, axis=0)
    box.masks = np.roll(box.masks, -leg, axis=0)
    box.levels[0] = box.gas
    box.masks[0] = True
    q, g = trajectory(box)
    return dict(
        m=box.mass,
        number=np.full(len(box.mass), 1 / box.volume),
        levels=box.levels,
        mask=box.masks,
        q=q,
        g=g,
        t=box.t,
        ids=family,
        mix=mix,
        density=density,
        source=last,
        prior_ode=phys,
        prior_learned=learned,
        anchor=anchor,
    )


def run(a):
    """Write ODE continuations with the source of each starting state."""
    jobs = [
        (f, m, d, a.seed, a.model, a.steps)
        for f in range(a.first, a.first + a.families)
        for d in (1, 8)
        for m in (0, 1)
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        records = list(pool.map(sample, jobs, chunksize=2))
    np.savez_compressed(
        a.out, **{k: np.stack([r[k] for r in records]) for k in records[0]}
    )
    print(a.out, len(records), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--model", default="parent")
    for k, v in (
        ("first", 11401),
        ("families", 192),
        ("seed", 5021),
        ("steps", 4),
        ("workers", 4),
    ):
        p.add_argument("--" + k, type=int, default=v)
    run(p.parse_args())
