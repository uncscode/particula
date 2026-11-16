"""Full prepared forcing study with explicitly coupled adaptive condensation."""

from dataclasses import replace

import numpy as np
import warp as wp
from particula.execution.gpu_coupled import (
    GPUCondensationControls,
    GPUCoupledSimulation,
)
from particula.util.constants import GAS_CONSTANT
from scripts.gpu_studies.cases import radius_histogram, water_pressure
from scripts.gpu_studies.validation import mapped_inventory, state_hash


def frame(owner, step):
    """Save compact observations after complete coupled cycles."""
    data, snapshot = owner.diagnostics(), owner.snapshot()
    edges, histogram = radius_histogram(snapshot)
    particles = snapshot.particles
    mass = np.sum(particles.masses * particles.concentration[..., None], axis=1)
    return {
        "step": step,
        "time_s": owner.time,
        "temperature_K": data.temperature.tolist(),
        "water_kg_m3": data.gas_concentration[:, 2].tolist(),
        "water_saturation": data.saturation_ratio[:, 2].tolist(),
        "represented_number": particles.concentration.sum(axis=1).tolist(),
        "particle_mass_kg": mass.tolist(),
        "radius_edges_m": edges.tolist(),
        "histogram_number_m3": histogram.tolist(),
    }, snapshot


def run_once(case, initial, processes, *, rtol):
    """Retain every original forcing, dilution and observation boundary."""
    condensation, dilution = processes
    frames = []
    expected = mapped_inventory(initial)
    external = np.zeros_like(expected)
    max_error = 0.0
    controls = GPUCondensationControls(rtol=rtol, first_step=case.time_step)
    with GPUCoupledSimulation(
        initial,
        (dilution,),
        condensation.condensation_strategy,
        device="cuda:0",
        time_step=case.time_step,
        root_seed=case.seed,
        order="condensation-first",
        controls=controls,
    ) as owner:
        owner.capture_events()
        observation, _ = frame(owner, 0)
        frames.append(observation)
        for step in range(case.steps):
            if step % (case.steps // 120) == 0:
                phase = step / case.steps
                temperature = np.interp(
                    phase, [0, 1 / 3, 2 / 3, 1], [298.15, 275, 310, 298.15]
                ) + np.linspace(-4, 4, case.boxes)
                owner.update_environment(temperature=temperature)
                if step in (case.steps // 3, 2 * case.steps // 3):
                    observed = owner.diagnostics()
                    gas = observed.gas_concentration.copy()
                    humidity = 1.6 if step == case.steps // 3 else 0.8
                    gas[:, 2] = (
                        humidity
                        * water_pressure(temperature)
                        * initial.gas.molar_mass[2]
                        / (GAS_CONSTANT * temperature)
                    )
                    change = (
                        gas[:, 2] - observed.gas_concentration[:, 2]
                    ) * observed.volume
                    external[:, 0] += change
                    expected[:, 0] += change
                    owner.update_gas(gas)
            owner.step()
            expected *= np.exp(-0.02 * case.time_step)
            if (step + 1) % (case.steps // 12) == 0 or step + 1 == case.steps:
                observation, current = frame(owner, step + 1)
                frames.append(observation)
                actual = mapped_inventory(current)
                np.testing.assert_allclose(
                    actual, expected, rtol=2e-12, atol=1e-30
                )
                max_error = max(
                    max_error, float(np.max(np.abs(actual / expected - 1)))
                )
        final = owner.snapshot()
        work = {
            **vars(owner.statistics),
            "control_readbacks": owner.control_readbacks,
            "coupling_downloads": owner.coupling_downloads,
            "coupling_uploads": owner.coupling_uploads,
        }
    if not np.any(final.particles.masses != initial.particles.masses):
        raise AssertionError("no condensation response")
    return {
        "frames": frames,
        "result_hash": state_hash(final),
        "external_mass_kg_by_box_material": external.tolist(),
        "max_box_species_relative_error": max_error,
        "work": work,
    }, final


def run_adaptive_forcing(case, initial, processes):
    """Qualify the full trajectory and refinement on actual CUDA."""
    if not wp.is_cuda_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    if case.experiment != "forcing" or case.steps < 120 or case.steps % 120:
        raise ValueError("forcing needs 120 exact control intervals")
    report, coarse = run_once(case, initial, processes, rtol=1e-5)
    fine_case = replace(
        case, time_step=case.time_step / 2, steps=case.steps * 2
    )
    fine_report, fine = run_once(fine_case, initial, processes, rtol=1e-7)
    active = fine.particles.concentration > 0
    mass_error = float(
        np.max(
            np.abs(
                coarse.particles.masses.sum(axis=2)[active]
                / fine.particles.masses.sum(axis=2)[active]
                - 1
            )
        )
    )
    gas_error = float(
        np.max(
            np.abs(
                coarse.gas.concentration[:, 2] / fine.gas.concentration[:, 2]
                - 1
            )
        )
    )
    if mass_error >= 0.01 or gas_error >= 0.001:
        raise AssertionError(
            f"forcing refinement failed: mass={mass_error}, gas={gas_error}"
        )
    report["refinement"] = {
        "coarse_time_step_s": case.time_step,
        "fine_time_step_s": fine_case.time_step,
        "coarse_rtol": 1e-5,
        "fine_rtol": 1e-7,
        "max_particle_total_mass_relative_change": mass_error,
        "max_water_vapor_relative_change": gas_error,
        "fine_inventory_relative_error": fine_report[
            "max_box_species_relative_error"
        ],
        "fine_work": fine_report["work"],
        "same_physical_duration_and_forcing": True,
    }
    return report
