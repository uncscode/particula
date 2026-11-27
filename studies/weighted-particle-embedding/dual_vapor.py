"""Two finite vapor reservoirs over particles with matched dry spectra."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from dataset import draw
from dual_materials import (
    DENSITY,
    MOLAR,
    NAMES,
    SATURATED,
    SURFACE_DENSITY,
    SURFACE_TENSION,
)
from particula import Aerosol
from particula.dynamics import CondensationIsothermal, MassCondensation
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import (
    ConstantVaporPressureStrategy,
    EnvironmentData,
    GasData,
)
from particula.particles import ActivityIdealMolar, ParticleData
from particula.particles.surface_strategies import SurfaceStrategyVolume
from pulse import INTERVAL, VOLUME, make_box
from vapor_properties import materials

OFFSETS = {
    "train": 10000000,
    "validation": 11000000,
    "test": 12000000,
    "challenge": 13000000,
}


def process(properties=None):
    """Both active vapors share the molar composition of all four materials."""
    spec = materials(2) if properties is None else properties
    return MassCondensation(
        CondensationIsothermal(
            molar_mass=spec.molar[: spec.species].copy(),
            diffusion_coefficient=spec.diffusivity.copy(),
            accommodation_coefficient=spec.accommodation.copy(),
            activity_strategy=ActivityIdealMolar(spec.molar.copy()),
            surface_strategy=SurfaceStrategyVolume(
                density=SURFACE_DENSITY, surface_tension=SURFACE_TENSION
            ),
            vapor_pressure_strategy=tuple(
                ConstantVaporPressureStrategy(x) for x in spec.pressure
            ),
            species_map=CondensationSpeciesMap(
                spec.names, tuple((s, s) for s in range(spec.species))
            ),
        )
    )


def initial(ids, bins, phase, challenge=False):
    """Generate the same particles for both phase schedules and both
    refinements.
    """
    group, wet, dry, organic0 = [], [], [], []
    for seed in ids:
        f = draw(int(seed), challenge)
        rng = np.random.default_rng([int(seed), 331])
        group.append(
            replace(
                f,
                composition_slope=rng.uniform(-0.15, 0.15),
                second_mode_fraction=rng.uniform(0.05, 0.95),
            )
        )
        hi = rng.uniform(0.45, 0.75) if challenge else rng.uniform(0.16, 0.45)
        lo = rng.uniform(0.001, 0.012)
        fraction = f.initial_water_ratio
        organic0.append(rng.uniform(0.04, 0.12) * fraction / 0.25)
        wet.append([f.wet_rh, hi if phase == "together" else lo])
        dry.append([f.dry_rh, lo if phase == "together" else hi])
    state, _ = make_box(group, bins)
    base, counts = state.particles.masses, state.particles.concentration
    org = np.repeat(organic0, 2)[:, None] * base[..., 1:].sum(-1)
    m = np.concatenate((base[..., :1], org[..., None], base[..., 1:]), -1)
    gas_levels = np.stack((wet, dry), 1) * SATURATED
    vapor = gas_levels[:, 0].repeat(2, 0)
    particles = ParticleData(
        masses=m,
        concentration=counts,
        density=DENSITY.copy(),
        volume=state.particles.volume,
        charge=np.zeros_like(counts),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        list(NAMES), MOLAR[:2].copy(), vapor.copy(), np.ones(2, dtype=bool)
    )
    bulk = (m * counts[..., None]).sum(1)
    np.testing.assert_allclose(bulk[::2], bulk[1::2], rtol=1e-14)
    return Aerosol(
        particles=particles,
        gas=gas,
        environment=EnvironmentData(
            state.environment.temperature.copy(),
            state.environment.pressure.copy(),
            vapor / SATURATED,
        ),
    ), gas_levels


def audit(volatile, gas, number, times):
    """Verify both closed reservoirs and dry-seed continuity."""
    if (
        not np.isfinite(volatile).all()
        or not np.isfinite(gas).all()
        or min(volatile.min(), gas.min()) < 0
    ):
        raise ValueError("Invalid volatile inventory")
    bulk = np.einsum("tbis,bi->tbs", volatile, number) / VOLUME
    resets = np.flatnonzero(np.diff(times) == 0) + 1
    for g, w in zip(np.split(gas, resets), np.split(bulk, resets), strict=True):
        np.testing.assert_allclose(g - g[0], w[0] - w, rtol=1e-8, atol=3e-15)
    np.testing.assert_array_equal(volatile[resets - 1], volatile[resets])
    return bulk


def schedule_inputs(levels, samples, reset_mask):
    """Validate pulse lengths and per-species reservoir changes."""
    lengths = np.broadcast_to(samples, (levels.shape[1],)).astype(int)
    mask = (
        np.ones_like(levels, dtype=bool) if reset_mask is None else reset_mask
    )
    if mask.shape != levels.shape or not mask[:, 0].all() or min(lengths) < 1:
        raise ValueError("Invalid partial-reset schedule")
    return lengths, mask


def integrate(
    state, levels, steps, samples, backend, reset_mask=None, properties=None
):
    """Record both vapors, their reservoirs and the fixed dry seeds."""
    spec = materials(2) if properties is None else properties
    lanes = spec.species
    proc = process(spec)
    number = state.particles.concentration.copy()
    mass0 = state.particles.masses.copy()
    lengths, mask = schedule_inputs(levels, samples, reset_mask)
    starts = np.r_[0, np.cumsum(lengths)]
    owner = None
    particle, gases, times = [], [], []
    try:
        if backend == "cuda":
            from dual_native_gpu import DirectDualCondensation

            owner = DirectDualCondensation(state, INTERVAL / steps, spec)
            owner.capture()
        for leg, length in enumerate(lengths):
            if leg:
                reset = np.where(
                    mask[:, leg].repeat(2, 0),
                    levels[:, leg].repeat(2, 0),
                    gases[-1],
                )
                if owner:
                    owner.update_gas(reset)
                else:
                    state.gas.concentration[:] = reset
            for j in range(length + 1):
                if j:
                    if owner:
                        owner.run(steps)
                    else:
                        for _ in range(steps):
                            proc.execute(
                                state, time_step=INTERVAL / steps, sub_steps=4
                            )
                snap = owner.snapshot() if owner else state
                np.testing.assert_array_equal(
                    snap.particles.masses[..., lanes:], mass0[..., lanes:]
                )
                np.testing.assert_array_equal(
                    snap.particles.concentration, number
                )
                particle.append(snap.particles.masses[..., :lanes].copy())
                gases.append(snap.gas.concentration.copy())
                times.append((starts[leg] + j) * INTERVAL)
    finally:
        if owner:
            owner.close()
    volatile, gas = np.stack(particle), np.stack(gases)
    bulk = audit(volatile, gas, number, times)
    return dict(
        mass0=mass0,
        counts=number,
        volatile=volatile,
        gas=gas,
        bulk=bulk,
        time=times,
        levels=levels,
        reset_mask=mask,
        volume=VOLUME,
        density=spec.density,
        molar_mass=spec.molar,
        steps=steps,
        backend=backend,
    )


def generate(
    root, split, phase, steps, families, bins, samples, batch, backend
):
    """Generate independent native trajectories under one phase schedule."""
    out = root / split / phase / f"s{steps}-b{bins}-t{samples}-{backend}"
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, families, batch):
        ids = OFFSETS[split] + np.arange(start, min(start + batch, families))
        path = out / f"families-{ids[0]}-{ids[-1]}.npz"
        if path.exists():
            with np.load(path) as d:
                np.testing.assert_array_equal(d["ids"], ids)
                if d["phase"] != phase or d["steps"] != steps:
                    raise ValueError("Existing dual-vapor reference changed")
            continue
        state, levels = initial(ids, bins, phase, split == "challenge")
        data = integrate(state, levels, steps, samples, backend)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data, ids=ids, phase=phase)
        tmp.replace(path)
        print(path, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("split", choices=OFFSETS)
    p.add_argument("phase", choices=("together", "opposed"))
    for name, v in (
        ("steps", 32),
        ("samples", 16),
        ("families", 64),
        ("bins", 16),
        ("batch", 8),
    ):
        p.add_argument("--" + name, type=int, default=v)
    p.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    generate(**vars(p.parse_args()))
