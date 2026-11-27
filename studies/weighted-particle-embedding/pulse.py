"""Native finite-reservoir mixing-state pairs. All masses and times use SI."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from particula import Aerosol
from particula.dynamics import CondensationIsothermal, MassCondensation
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import EnvironmentData, GasData
from particula.particles import ActivityKappaParameter, ParticleData
from particula.particles.surface_strategies import SurfaceStrategyVolume
from scripts.gpu_studies.humidity_campaign import pulse

DENSITY = np.array([1000.0, 1400.0, 1400.0])
MOLAR = np.array([pulse.M_WATER, 0.14, 0.10])
KAPPA = np.array([0.0, 0.03, 0.75])
INTERVAL = 0.0025
VOLUME = 1e-6


@dataclass(frozen=True)
class Family:
    """Specify a dry size spectrum, composition contrast and vapor pulse."""

    radius_um: float = 0.65
    number_m3: float = 1e11
    soluble: float = 0.50
    contrast: float = 0.90
    wet_rh: float = 0.90
    dry_rh: float = 0.35
    initial_water_ratio: float = 0.25
    log_width: float = 0.28
    rich_number_fraction: float = 0.5
    composition_slope: float = 0.0
    second_mode_fraction: float | None = None


def make_box(families, bins):
    """Build matched internal and segregated aerosols with one vapor lane."""
    masses, counts = [], []
    for f in families:
        dry_radius = (
            f.radius_um
            * 1e-6
            * np.exp(f.log_width * np.linspace(-1.6, 1.6, bins))
        )
        dry = np.repeat((4 * np.pi / 3) * dry_radius**3 * DENSITY[1], 2)
        axis = np.linspace(-1.0, 1.0, bins)
        size_share = np.full(bins, 1 / bins)
        if f.second_mode_fraction is not None:
            first = np.exp(-0.5 * ((axis + 0.63) / 0.23) ** 2)
            second = np.exp(-0.5 * ((axis - 0.63) / 0.23) ** 2)
            size_share = (
                (1 - f.second_mode_fraction) * first / first.sum()
                + f.second_mode_fraction * second / second.sum()
            )
        # Apply the same dry compositional gradient to both mixture types.
        average = f.soluble + f.composition_slope * axis
        rich = f.rich_number_fraction
        if not (0 < rich < 1 and np.all((average > 0) & (average < 1))):
            raise ValueError("Invalid class weights or dry material fractions")
        for contrast in (0.0, f.contrast):
            delta = contrast * np.minimum(
                1 - average, average * (1 - rich) / rich
            )
            share = np.column_stack(
                [average - delta * rich / (1 - rich), average + delta]
            ).ravel()
            masses.append(
                np.stack(
                    (
                        dry * f.initial_water_ratio,
                        dry * (1 - share),
                        dry * share,
                    ),
                    axis=-1,
                )
            )
            counts.append(
                (
                    size_share[:, None]
                    * np.array([1 - rich, rich])[None, :]
                    * (f.number_m3 * VOLUME)
                ).ravel()
            )
    masses, counts = np.asarray(masses), np.asarray(counts)
    inventory = np.einsum("bi,bis->bs", counts, masses) / VOLUME
    np.testing.assert_allclose(inventory[::2], inventory[1::2], rtol=4e-15)
    np.testing.assert_allclose(
        masses[::2, :, 1:].sum(-1), masses[1::2, :, 1:].sum(-1), rtol=4e-15
    )
    rh = np.repeat([f.wet_rh for f in families], 2)
    particles = ParticleData(
        masses=masses,
        concentration=counts,
        volume=np.full(len(rh), VOLUME),
        density=DENSITY.copy(),
        charge=np.zeros_like(counts),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        ["water"],
        MOLAR[:1].copy(),
        np.array([[pulse.gas_water(r)] for r in rh]),
        np.array([True]),
    )
    environment = EnvironmentData(
        np.full(len(rh), pulse.TEMPERATURE),
        np.full(len(rh), 101325.0),
        rh[:, None],
    )
    return Aerosol(
        particles=particles, gas=gas, environment=environment
    ), make_process()


def make_process():
    """Construct the common three-material water-exchange process."""
    activity = ActivityKappaParameter(
        kappa=KAPPA, density=DENSITY, molar_mass=MOLAR, water_index=0
    )
    return MassCondensation(
        CondensationIsothermal(
            molar_mass=MOLAR[:1].copy(),
            diffusion_coefficient=2.4e-5,
            accommodation_coefficient=0.2,
            activity_strategy=activity,
            surface_strategy=SurfaceStrategyVolume(
                density=1000.0, surface_tension=0.072
            ),
            vapor_pressure_strategy=pulse.WATER,
            species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
        )
    )


def check_trajectory(mass, gas, counts, initial, samples):
    """Check closed-leg budgets, dry inventories and finite positive state."""
    np.testing.assert_equal(
        mass[:, :, :, 1:],
        np.broadcast_to(initial[:, :, 1:], mass[:, :, :, 1:].shape),
    )
    water = np.einsum("tbi,bi->tb", mass[:, :, :, 0], counts) / VOLUME
    for g, p in zip(np.split(gas, 2), np.split(water, 2), strict=True):
        np.testing.assert_allclose(g - g[0], p[0] - p, rtol=1e-8, atol=1e-15)
    np.testing.assert_array_equal(mass[samples], mass[samples + 1])
    if not all(np.isfinite(a).all() and np.all(a >= 0) for a in (mass, gas)):
        raise ValueError("Invalid native trajectory")
    return water


def simulate(families, bins, steps, samples, backend):
    """Record independent closed legs around one imposed vapor replacement."""
    state, process = make_box(families, bins)
    return integrate(state, process, families, steps, samples, backend)


def integrate(state, process, families, steps, samples, backend):
    """Run the common schedule from a native or sectional initial state."""
    dt = INTERVAL / steps
    counts = state.particles.concentration.copy()
    initial = state.particles.masses.copy()
    reset = np.repeat(
        [[pulse.gas_water(f.dry_rh)] for f in families], 2, axis=0
    )
    owner = None
    mass, gas, times = [], [], []
    try:
        if backend == "cuda":
            owner = process.prepare(state, time_step=dt, device="cuda:0")
            owner.capture()
        for leg in range(2):
            if leg:
                if owner:
                    owner.update_gas(reset)
                else:
                    state.gas.concentration[:] = reset
            for sample in range(samples + 1):
                if sample:
                    if owner:
                        owner.run(steps)
                    else:
                        for _ in range(steps):
                            process.execute(state, time_step=dt, sub_steps=4)
                snap = owner.snapshot() if owner else state
                mass.append(snap.particles.masses.copy())
                gas.append(snap.gas.concentration[:, 0].copy())
                times.append((sample + leg * samples) * INTERVAL)
    finally:
        if owner:
            owner.close()
    mass, gas = np.array(mass), np.array(gas)
    water = check_trajectory(mass, gas, counts, initial, samples)
    return dict(
        mass=mass,
        gas=gas,
        counts=counts,
        water=water,
        time=np.array(times),
        external_water=gas[samples + 1] - gas[samples],
        density=DENSITY,
        volume=VOLUME,
        families=json.dumps([asdict(f) for f in families]),
        steps=steps,
        backend=backend,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    result = simulate(
        [Family()], args.bins, args.steps, args.samples, args.backend
    )
    np.savez_compressed(args.output, **result)
    print(
        "External minus internal water (kg/m3), wet and dry endpoints",
        np.diff(result["water"][[args.samples, -1]], axis=1).ravel().tolist(),
    )
