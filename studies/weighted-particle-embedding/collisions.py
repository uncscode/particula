"""Particle-resolved Brownian histories with optional simultaneous water
exchange.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from particula.dynamics import BrownianCoagulationStrategy, Coagulation
from pulse import DENSITY, Family, make_box, pulse


def configurations(ids, mode):
    """Draw a dry size and composition family at two number densities."""
    families = []
    for key in ids:
        rng = np.random.default_rng(key)
        base = Family(
            radius_um=rng.uniform(0.045, 0.10),
            number_m3=10 ** rng.uniform(14.05, 14.50),
            soluble=rng.uniform(0.22, 0.7),
            contrast=rng.uniform(0.4, 0.95),
            wet_rh=rng.uniform(0.76, 0.91),
            dry_rh=rng.uniform(0.38, 0.54),
            initial_water_ratio=rng.uniform(0.16, 0.3),
            log_width=rng.uniform(0.2, 0.45),
        )
        families.extend([base, replace(base, number_m3=2 * base.number_m3)])
    return families


def histories(ids, mode, slots, samples, steps, seed):
    """Run the paired Brownian ensemble and verify its phase inventories."""
    families = configurations(ids, mode)
    state, condensation = make_box(families, slots // 2)
    coagulation = Coagulation(BrownianCoagulationStrategy("particle_resolved"))
    process = (
        coagulation if mode == "coagulation" else condensation | coagulation
    )
    interval = 0.125
    dt = interval / steps
    names = tuple(
        f"family-{key}-density-{density}-mix-{mix}-rep-{seed}"
        for key in ids
        for density in range(2)
        for mix in range(2)
    )
    initial = state.particles
    initial_mass = (
        np.einsum("bi,bis->bs", initial.concentration, initial.masses)
        / initial.volume[:, None]
    )
    mass, counts, gas, time = [], [], [], []
    with process.prepare(
        state,
        device="cuda:0",
        time_step=dt,
        root_seed=seed,
        logical_box_ids=names,
    ) as owner:
        owner.capture()
        for leg in range(2):
            if leg and mode == "coupled":
                owner.update_gas(
                    np.repeat(
                        [[pulse.gas_water(f.dry_rh)] for f in families],
                        2,
                        axis=0,
                    )
                )
            for frame in range(samples + 1):
                if frame:
                    owner.run(steps)
                current = owner.snapshot()
                mass.append(current.particles.masses.copy())
                counts.append(current.particles.concentration.copy())
                gas.append(current.gas.concentration[:, 0].copy())
                time.append((leg * samples + frame) * interval)
    mass, counts, gas = map(np.array, (mass, counts, gas))
    material = (
        np.einsum("tbi,tbis->tbs", counts, mass) / initial.volume[None, :, None]
    )
    np.testing.assert_allclose(
        material[:, :, 1:],
        np.broadcast_to(initial_mass[None, :, 1:], material[:, :, 1:].shape),
        rtol=2e-13,
    )
    for g, m in zip(
        np.split(gas, 2), np.split(material[:, :, 0], 2), strict=True
    ):
        np.testing.assert_allclose(g - g[0], m[0] - m, rtol=1e-8, atol=2e-15)
    if mode == "coagulation":
        np.testing.assert_allclose(
            material[:, :, 0],
            np.broadcast_to(initial_mass[:, 0], material[:, :, 0].shape),
            rtol=2e-13,
        )
    n = counts.sum(-1)
    if (np.diff(n, axis=0) > 0).any() or not (n[-1] < n[0]).any():
        raise ValueError(
            "Expected non-increasing number and at least one collision"
        )
    if not all(
        np.isfinite(a).all() and np.all(a >= 0) for a in (mass, counts, gas)
    ):
        raise ValueError("Invalid native coagulation state")
    return dict(
        mass=mass,
        counts=counts,
        gas=gas,
        water=material[:, :, 0],
        time=np.array(time),
        ids=np.repeat(ids, 2),
        replicate=seed,
        volume=initial.volume[0],
        density=DENSITY,
        mode=mode,
        steps=steps,
        backend="cuda",
        families=json.dumps([asdict(f) for f in families]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("mode", choices=["coagulation", "coupled"])
    p.add_argument("split", choices=["train", "validation", "test"])
    for k, default in (
        ("groups", 8),
        ("batch", 4),
        ("slots", 64),
        ("samples", 8),
        ("steps", 8),
        ("seed", 71),
        ("first", 0),
    ):
        p.add_argument("--" + k, type=int, default=default)
    a = p.parse_args()
    if a.slots < 2 or a.slots % 2:
        p.error("slots must be a positive even number")
    out = a.root / a.mode / a.split
    out.mkdir(parents=True, exist_ok=True)
    offset = {"train": 500000, "validation": 600000, "test": 700000}[a.split]
    for first in range(a.first, a.first + a.groups, a.batch):
        ids = list(
            range(
                offset + first,
                offset + min(first + a.batch, a.first + a.groups),
            )
        )
        path = out / (
            f"families-{ids[0]}-{ids[-1]}-slots{a.slots}"
            f"-steps{a.steps}-t{a.samples}-seed{a.seed}.npz"
        )
        if path.exists():
            continue
        data = histories(ids, a.mode, a.slots, a.samples, a.steps, a.seed)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(path)
        c = data["counts"].sum(-1)
        print(
            path.name,
            "mean number remaining",
            float((c[-1] / c[0]).mean()),
            flush=True,
        )
