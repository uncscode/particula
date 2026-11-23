"""Descend in humidity with distinct native particles and direct AIOMFAC."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from particula import Aerosol
from particula.dynamics import CondensationIsothermal, MassCondensation
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData, SurfaceStrategyVolume
from particula.particles.activity_factories import ActivityFactory
from particula.util.constants import GAS_CONSTANT
from particula_aiomfac import MixtureSpec

GROUPS = {
    "water": {16: 1},
    "2-propanol": {141: 2, 151: 1, 153: 1},
    "NaCl": {202: 1, 242: 1},
    "KCl": {203: 1, 242: 1},
}
T = 298.15
P = 101325.0
P_SAT = 3169.0
WATER_MW = 0.01801528
DENSITY = np.array([1000.0, 785.0, 2165.0, 1984.0])
SIGMA = 0.072
VOLUME = 1e-5
C_SAT = P_SAT * WATER_MW / (GAS_CONSTANT * T)
RH_START = 0.90
RH_END = 0.82
DURATION = 0.192
SEED = 2012


def initial_state(n_particles):
    """Return the factory activity, native box and condensation process."""
    if n_particles < 1:
        raise ValueError("at least one particle is required")
    activity = ActivityFactory().get_strategy(
        "aiomfac",
        dict(
            mixture=MixtureSpec(GROUPS),
            material_mapping=dict(enumerate(GROUPS)),
            phase_mode="stable",
        ),
    )
    rng = np.random.default_rng(SEED)
    # Interleaved draws preserve the first IDs in reduced runs.
    draws = rng.random((n_particles, 3))
    salt = 0.19 + 0.17 * draws[:, 0]
    potassium = 0.15 + 0.55 * draws[:, 1]
    total_dry = 1e-14 * (0.7 + 0.6 * draws[:, 2])
    dry = total_dry[:, None] * np.column_stack(
        (1.0 - salt, salt * (1.0 - potassium), salt * potassium)
    )
    mass = np.column_stack((total_dry, dry))

    def surface_rh(water):
        mass[:, 0] = water
        radius = np.cbrt(3 * (mass / DENSITY).sum(axis=1) / (4 * np.pi))
        activity_water = activity.prepared.evaluate(mass, T).neutral_activities[
            :, 0
        ]
        return activity_water * np.exp(
            2 * SIGMA * WATER_MW / (DENSITY[0] * GAS_CONSTANT * T * radius)
        )

    # Initialize a homogeneous solution in equilibrium with the wet reservoir.
    # The stable solve at t=0 must certify one liquid in every row.
    lo, hi = total_dry * 2.0, total_dry * 30.0
    if (surface_rh(lo) > RH_START).any() or (surface_rh(hi) < RH_START).any():
        raise RuntimeError("could not bracket the wet starting compositions")
    for _ in range(44):
        mid = (lo + hi) / 2.0
        low = surface_rh(mid) < RH_START
        lo = np.where(low, mid, lo)
        hi = np.where(low, hi, mid)
    mass[:, 0] = (lo + hi) / 2
    box = Aerosol(
        particles=ParticleData(
            masses=mass[None],
            concentration=np.ones((1, n_particles)),
            charge=np.zeros((1, n_particles)),
            density=DENSITY.copy(),
            volume=np.array([VOLUME]),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["water"],
            np.array([WATER_MW]),
            np.array([[C_SAT * RH_START]]),
            np.array([True]),
        ),
        environment=EnvironmentData([T], [P], [[RH_START]]),
    )
    strategy = CondensationIsothermal(
        molar_mass=np.array([WATER_MW]),
        diffusion_coefficient=2.5e-5,
        accommodation_coefficient=1.0,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(P_SAT),
        activity_strategy=activity,
        surface_strategy=SurfaceStrategyVolume(
            density=DENSITY[0], surface_tension=SIGMA
        ),
        species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
        update_gases=True,
    )
    return activity, box, MassCondensation(strategy)


def run_ramp(
    n_particles=1000,
    device="cuda:0",
    *,
    intervals=32,
    substeps=1,
    particle_ids=None,
):
    """Evolve every occupied slot and return JSON-serializable observations.

    Each forcing change replaces the box vapor concentration; the box then
    evolves conservatively for one native condensation timestep. RH at the
    observation is computed from the final gas concentration. A crossing
    bracket spans adjacent observations, not a root of a bulk phase diagram.
    """
    import warp as wp

    if intervals < 1 or substeps < 1 or not wp.get_device(device).is_cuda:
        raise ValueError(
            "choose a positive interval count and a real CUDA device"
        )
    start = perf_counter()
    provider, box, process = initial_state(n_particles)
    ids = (
        np.arange(n_particles)
        if particle_ids is None
        else np.asarray(particle_ids, dtype=int)
    )
    if (
        ids.ndim != 1
        or len(ids) == 0
        or len(np.unique(ids)) != len(ids)
        or ids.min() < 0
        or ids.max() >= n_particles
    ):
        raise ValueError(
            "choose unique particle IDs within the initialized population"
        )
    # Subset convergence runs keep unit particle weights and the original
    # number density by scaling only the observation volume.
    volume = VOLUME * len(ids) / n_particles
    box.particles.masses = box.particles.masses[:, ids].copy()
    box.particles.concentration = box.particles.concentration[:, ids].copy()
    box.particles.charge = box.particles.charge[:, ids].copy()
    box.particles.volume[:] = volume
    n_particles = len(ids)
    sampling = np.linspace(0.0, DURATION, intervals + 1)
    forcing = np.linspace(RH_START, RH_END, intervals + 1)
    binding = provider.bind_device(box.particles.masses.shape, (0,), device)
    counts = wp.ones((1, n_particles), dtype=wp.float64, device=device)
    tk = wp.array([T], dtype=wp.float64, device=device)
    invalid = wp.zeros(1, dtype=wp.int32, device=device)

    def diagnose(current):
        # This independent observer uses the production stable kernel on the
        # accepted masses. Transport evolves its own particle inventories.
        binding.evaluate(
            wp.array(current[None], device=device), counts, tk, invalid
        )
        codes = binding.status.numpy()[0]
        if (codes < 0).any() or invalid.numpy()[0]:
            slots = np.flatnonzero(codes < 0)
            raise RuntimeError(
                f"unresolved accepted phases for IDs {slots}: {codes[slots]}"
            )
        return np.where(codes == 2, 2, 1)

    initial = box.particles.masses[0].copy()
    water = [initial[:, 0].copy()]
    phase = [diagnose(initial)]
    if np.any(phase[0] != 1):
        raise RuntimeError(
            "wet initialization requires a stable homogeneous liquid"
        )
    rh, external, residual = [RH_START], [0.0], [0.0]
    gas = C_SAT * RH_START
    budget0 = initial[:, 0].sum() + gas * volume
    dry_error = np.zeros(n_particles)
    with process.prepare(
        box, time_step=DURATION / (intervals * substeps), device=device
    ) as owner:
        for target in forcing[1:]:
            assigned = C_SAT * target
            external.append(external[-1] + (assigned - gas) * volume)
            owner.update_gas(np.array([assigned]))
            owner.run(substeps)
            result = owner.snapshot()
            current = result.particles.masses[0]
            if np.any(result.particles.concentration[0] != 1):
                raise RuntimeError("particle ID occupancy changed")
            dry_error = np.maximum(
                dry_error, np.max(abs(current[:, 1:] - initial[:, 1:]), axis=1)
            )
            gas = float(result.gas.concentration[0, 0])
            water.append(current[:, 0].copy())
            phase.append(diagnose(current))
            rh.append(gas / C_SAT)
            residual.append(
                current[:, 0].sum() + gas * volume - budget0 - external[-1]
            )
    if dry_error.any() or max(abs(np.array(residual))) > 2e-12 * budget0:
        raise RuntimeError("native material inventory balance failed")

    phase = np.array(phase)
    first = np.argmax(phase == 2, axis=0)
    observed = (phase == 2).any(axis=0)
    crossing = []
    for event, k in zip(observed, first, strict=True):
        crossing.append([int(k - 1), int(k)] if event and k else None)
    return dict(
        materials=list(GROUPS),
        particle_id=ids.tolist(),
        dry_mass_kg=initial[:, 1:].tolist(),
        initial_water_kg=initial[:, 0].tolist(),
        water_kg=np.array(water).tolist(),
        phase_count=phase.tolist(),
        time_s=sampling.tolist(),
        rh=rh,
        imposed_rh=forcing.tolist(),
        onset_sample_bracket=crossing,
        onset_censor=["bracketed" if event else "right" for event in observed],
        water_transfer_kg=(water[-1] - water[0]).tolist(),
        maximum_dry_mass_error_kg=dry_error.tolist(),
        gas_water_kg=(np.array(rh) * C_SAT * volume).tolist(),
        cumulative_imposed_water_kg=external,
        balance_residual_kg=residual,
        temperature_K=T,
        pressure_Pa=P,
        surface_tension_N_m=SIGMA,
        material_density_kg_m3=DENSITY.tolist(),
        kelvin_density_kg_m3=float(DENSITY[0]),
        box_volume_m3=volume,
        substeps_per_interval=substeps,
        rng_seed=SEED,
        wall_seconds=perf_counter() - start,
        backend=str(wp.get_device(device).name),
    )


def refine_ramp(device="cuda:0"):
    """Refine transport for fixed IDs under identical gas replacements."""
    selected = np.linspace(0, 999, 16, dtype=int)
    cases = [
        run_ramp(device=device, particle_ids=selected, substeps=count)
        for count in (1, 2, 4)
    ]
    finest = cases[-1]
    summaries = []
    for case in cases:
        masses = np.asarray(case["water_kg"])
        fine_masses = np.asarray(finest["water_kg"])
        onset = case["onset_sample_bracket"]
        fine_onset = finest["onset_sample_bracket"]
        shifts = [
            a[1] - b[1]
            for a, b in zip(onset, fine_onset, strict=True)
            if a and b
        ]
        summaries.append(
            dict(
                substeps=case["substeps_per_interval"],
                wall_seconds=case["wall_seconds"],
                maximum_relative_water_mass_difference=float(
                    np.max(np.abs(masses - fine_masses) / fine_masses)
                ),
                maximum_ambient_rh_difference=float(
                    np.max(np.abs(np.array(case["rh"]) - finest["rh"]))
                ),
                maximum_onset_difference_in_intervals=max(
                    map(abs, shifts), default=0
                ),
                onset_sample_bracket=onset,
                onset_censor=case["onset_censor"],
                maximum_abs_balance_residual_kg=max(
                    map(abs, case["balance_residual_kg"])
                ),
            )
        )
    return dict(
        particle_id=selected.tolist(),
        time_s=finest["time_s"],
        imposed_rh=finest["imposed_rh"],
        box_volume_m3=finest["box_volume_m3"],
        cases=summaries,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--intervals", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement", action="store_true")
    args = parser.parse_args()
    result = (
        refine_ramp(args.device)
        if args.refinement
        else run_ramp(args.particles, args.device, intervals=args.intervals)
    )
    args.output.write_text(
        json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
    )
    if not args.refinement:
        print(
            f"{args.particles} particles, {len(result['time_s'])} samples, "
            f"{sum(c == 'bracketed' for c in result['onset_censor'])} onsets; "
            f"{result['wall_seconds']:.1f} s"
        )
