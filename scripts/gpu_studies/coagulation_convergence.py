"""Bimodal native sampling versus an independent continuous-time CPU chain.

Run explicitly; this is a scientific qualification study, not a timing test.
The reference recomputes the CPU kernel matrix after every Gillespie event.
GPU steps use the tiled sampler and explicitly inspect admission between steps.
Both represent unit counts at the same physical number density.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
from particula.dynamics import (
    BrownianCoagulationStrategy,
    CombineCoagulationStrategy,
    SedimentationCoagulationStrategy,
    TurbulentShearCoagulationStrategy,
)
from particula.gpu import to_warp_particle_data
from particula.gpu.kernels.coagulation import (
    CoagulationMechanismConfig,
    coagulation_preflight_gpu,
    coagulation_step_gpu,
)
from particula.particles import ParticleData

TEMPERATURE = 298.15
PRESSURE = 101325.0
DENSITY = 1000.0
NUMBER_DENSITY = 1e10
DISSIPATION = 1e-3
FLUID_DENSITY = 1.2


def population(slots, boxes=1):
    """Exactly resolve a 7:1 small/collector number ratio in every box."""
    radii = np.concatenate(
        (np.full(7 * slots // 8, 1e-7), np.full(slots // 8, 4e-6))
    )
    mass = 4 * np.pi / 3 * DENSITY * radii**3
    return ParticleData(
        np.broadcast_to(mass[None, :, None], (boxes, slots, 1)).copy(),
        np.ones((boxes, slots)),
        np.zeros((boxes, slots)),
        np.array([DENSITY]),
        np.full(boxes, slots / NUMBER_DENSITY),
        distribution_type="particle_resolved",
    )


def observables(masses, initial):
    """Number and surface area fractions; empty slots contribute nothing."""
    active = masses > 0
    radii = np.cbrt(3 * masses / (4 * np.pi * DENSITY))
    initial_radii = np.cbrt(3 * initial / (4 * np.pi * DENSITY))
    return np.array(
        [
            active.sum() / initial.size,
            np.count_nonzero(active & (radii < 0.5e-6)) / initial.size,
            np.sum(radii**2) / np.sum(initial_radii**2),
        ]
    )


def summarize(samples):
    """Retain uncertainty of independent realizations, not particle scatter."""
    return {
        "mean": np.mean(samples, axis=0).tolist(),
        "standard_error": (
            np.std(samples, axis=0, ddof=1) / np.sqrt(len(samples))
        ).tolist(),
    }


def cpu_reference(slots, ensembles, duration):
    """Exact event times for the finite-population CPU pair-rate model."""
    strategy = CombineCoagulationStrategy(
        [
            BrownianCoagulationStrategy("particle_resolved"),
            SedimentationCoagulationStrategy("particle_resolved"),
            TurbulentShearCoagulationStrategy(
                "particle_resolved", DISSIPATION, FLUID_DENSITY
            ),
        ]
    )
    initial = population(slots)
    original = initial.masses[0, :, 0]
    samples, events = [], []
    for realization in range(ensembles):
        random = np.random.default_rng(9187 + 1009 * slots + realization)
        masses = original.copy()
        elapsed = 0.0
        count = 0
        while masses.size > 1:
            local = ParticleData(
                masses[None, :, None],
                np.ones((1, masses.size)),
                np.zeros((1, masses.size)),
                np.array([DENSITY]),
                initial.volume.copy(),
                distribution_type="particle_resolved",
            )
            matrix = strategy.kernel(local, TEMPERATURE, PRESSURE)
            first, second = np.triu_indices(masses.size, 1)
            hazards = matrix[first, second] / initial.volume[0]
            total = hazards.sum()
            if total <= 0:
                break
            elapsed += random.exponential(1 / total)
            if elapsed > duration:
                break
            pair = min(
                int(
                    np.searchsorted(np.cumsum(hazards), random.random() * total)
                ),
                hazards.size - 1,
            )
            masses[first[pair]] += masses[second[pair]]
            masses = np.delete(masses, second[pair])
            count += 1
        np.testing.assert_allclose(
            masses.sum(), original.sum(), rtol=2e-14, atol=0
        )
        samples.append(observables(masses, original))
        events.append(count)
    return {**summarize(samples), "accepted_events": int(np.sum(events))}


def inputs(slots):
    """Freeze the same physical three-mechanism model on the device."""
    return dict(
        temperature=TEMPERATURE,
        pressure=PRESSURE,
        max_collisions=slots // 2,
        mechanism_config=CoagulationMechanismConfig(
            ("brownian", "sedimentation_sp2016", "turbulent_shear_st1956")
        ),
        turbulent_dissipation=DISSIPATION,
        fluid_density=FLUID_DENSITY,
    )


def device_ensemble(slots, ensembles, duration, nominal_step, device):
    """Explicitly advance every requested interval; never discard demand."""
    initial = population(slots, ensembles)
    particles = to_warp_particle_data(initial, device=device)
    rng = wp.zeros(ensembles, dtype=wp.uint32, device=device)
    elapsed, steps, accepted = 0.0, 0, 0
    minimum_step = nominal_step
    while elapsed < duration:
        report = coagulation_preflight_gpu(
            particles, time_step=1.0, **inputs(slots)
        )
        positive = report["max_time_step"][report["max_time_step"] > 0]
        resource_step = (
            0.9 * float(positive.min()) if positive.size else nominal_step
        )
        step = min(nominal_step, resource_step)
        if duration - elapsed <= step * (1 + 1e-12):
            step = duration - elapsed
        if step <= 0:
            raise AssertionError("nonpositive explicit study timestep")
        _, _, collisions = coagulation_step_gpu(
            particles,
            time_step=step,
            rng_states=rng,
            initialize_rng=steps == 0,
            rng_seed=711,
            **inputs(slots),
        )
        mass = particles.masses.numpy()[..., 0]
        counts = particles.concentration.numpy()
        np.testing.assert_allclose(
            (mass * counts).sum(axis=1),
            initial.masses[..., 0].sum(axis=1),
            rtol=2e-13,
            atol=0,
        )
        np.testing.assert_array_equal(particles.charge.numpy(), 0)
        accepted += int(collisions.numpy().sum())
        elapsed += step
        minimum_step = min(minimum_step, step)
        steps += 1
    np.testing.assert_equal(
        int(np.count_nonzero(counts)), ensembles * slots - accepted
    )
    samples = [observables(row, initial.masses[0, :, 0]) for row in mass]
    return {
        **summarize(samples),
        "steps": steps,
        "nominal_step_s": nominal_step,
        "minimum_step_s": minimum_step,
        "accepted_events": accepted,
    }


def run(device="cuda:0", ensembles=128):
    """Compare timestep refinement at three population sizes."""
    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("Native CUDA is required; no fallback")
    base = to_warp_particle_data(population(32), device=device)
    bound = float(
        coagulation_preflight_gpu(base, time_step=1.0, **inputs(32))[
            "max_time_step"
        ][0]
    )
    duration = 2 * bound
    rows = []
    for slots in (32, 64, 128):
        reference = cpu_reference(slots, ensembles, duration)
        refinements = []
        for divisor in (4, 8, 16):
            result = device_ensemble(
                slots, ensembles, duration, bound / divisor, device
            )
            difference = np.asarray(result["mean"]) - reference["mean"]
            uncertainty = np.hypot(
                result["standard_error"], reference["standard_error"]
            )
            result["reference_difference"] = difference.tolist()
            result["combined_standard_error"] = uncertainty.tolist()
            refinements.append(result)
        # Predeclared finite-ensemble agreement criterion at the finest step.
        fine = refinements[-1]
        tolerance = 5 * np.asarray(fine["combined_standard_error"]) + 1 / (
            ensembles * slots
        )
        if not np.all(np.abs(fine["reference_difference"]) <= tolerance):
            raise AssertionError((slots, fine))
        if reference["accepted_events"] <= 0 or not all(
            row["accepted_events"] > 0 for row in refinements
        ):
            raise AssertionError("convergence study requires actual collisions")
        rows.append(
            {
                "particles": slots,
                "cpu_gillespie": reference,
                "refinements": refinements,
            }
        )
        print(
            f"Validated {slots} particles, {ensembles} realizations", flush=True
        )
    return {
        "device": device,
        "ensembles": ensembles,
        "duration_s": duration,
        "observables": [
            "number_fraction",
            "small_particle_fraction",
            "surface_area_fraction",
        ],
        "number_density_m3": NUMBER_DENSITY,
        "small_radius_m": 1e-7,
        "collector_radius_m": 4e-6,
        "small_to_collector_ratio": 7,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ensembles", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.device, args.ensembles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
