"""Compose direct Warp process steps with one explicit transfer boundary.

This example keeps conversion, device selection, sidecars, RNG state,
synchronization, and the final CPU checkpoint under caller control. It is not
a scheduler, backend selector, resident loop, high-level Runnable, or CPU
fallback.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData

_FORCE_NO_WARP_ENV = "PARTICULA_EXAMPLE_FORCE_NO_WARP"


@dataclass
class ExampleRun:
    """Store deterministic output and optional final CPU/GPU-owned state.

    Every optional field is ``None`` when Warp is unavailable or disabled.
    """

    output: list[str]
    particle_data: ParticleData | None = None
    gas_data: GasData | None = None
    environment_data: EnvironmentData | None = None
    mass_transfer: Any | None = None
    collision_pairs: Any | None = None
    n_collisions: Any | None = None
    coagulation_rng: Any | None = None
    wall_rng: Any | None = None
    nucleation_diagnostics: Any | None = None


def _build_cpu_state() -> tuple[
    ParticleData, GasData, EnvironmentData, list[str]
]:
    """Create the deterministic B=1, N=4, S=2 direct-process fixture.

    Returns:
        CPU particle, gas, and environment containers plus copied gas names.
    """
    particle_data = ParticleData(
        masses=np.array(
            [[[1.0e-18, 2.0e-18], [0.0, 0.0], [3.0e-18, 0.0], [0.0, 0.0]]],
            dtype=np.float64,
        ),
        concentration=np.array([[2.0, 0.0, 5.0, 0.0]], dtype=np.float64),
        charge=np.array([[1.0, 0.0, -2.0, 0.0]], dtype=np.float64),
        density=np.array([1000.0, 1500.0], dtype=np.float64),
        volume=np.array([1.0e-6], dtype=np.float64),
    )
    gas_data = GasData(
        name=["water", "organic"],
        molar_mass=np.array([0.018, 0.098], dtype=np.float64),
        concentration=np.array([[1.0e-9, 2.0e-9]], dtype=np.float64),
        partitioning=np.array([True, True], dtype=np.bool_),
    )
    environment_data = EnvironmentData(
        temperature=np.array([298.15], dtype=np.float64),
        pressure=np.array([101325.0], dtype=np.float64),
        saturation_ratio=np.ones((1, 2), dtype=np.float64),
    )
    return particle_data, gas_data, environment_data, list(gas_data.name)


def _warp_enabled() -> bool:
    """Return whether optional Warp execution can be loaded without GPU imports."""
    if os.getenv(_FORCE_NO_WARP_ENV) == "1":
        return False
    try:
        importlib.import_module("warp")
    except ImportError:
        return False
    return True


def _load_enabled_runtime() -> SimpleNamespace | None:
    """Lazily load the optional GPU package, public steps, and concrete records."""
    wp = importlib.import_module("warp")
    gpu = importlib.import_module("particula.gpu")
    if not gpu.WARP_AVAILABLE:
        return None
    kernels = importlib.import_module("particula.gpu.kernels")
    condensation = importlib.import_module("particula.gpu.kernels.condensation")
    thermodynamics = importlib.import_module(
        "particula.gpu.kernels.thermodynamics"
    )
    coagulation = importlib.import_module("particula.gpu.kernels.coagulation")
    wall_loss = importlib.import_module("particula.gpu.kernels.wall_loss")
    exhaustion = importlib.import_module("particula.gpu.kernels.exhaustion")
    nucleation = importlib.import_module("particula.gpu.kernels.nucleation")
    return SimpleNamespace(
        wp=wp,
        gpu=gpu,
        condensation_step_gpu=kernels.condensation_step_gpu,
        coagulation_step_gpu=kernels.coagulation_step_gpu,
        dilution_step_gpu=kernels.dilution_step_gpu,
        wall_loss_step_gpu=kernels.wall_loss_step_gpu,
        nucleation_step_gpu=kernels.nucleation_step_gpu,
        CondensationScratchBuffers=condensation.CondensationScratchBuffers,
        ThermodynamicsConfig=thermodynamics.ThermodynamicsConfig,
        CoagulationMechanismConfig=coagulation.CoagulationMechanismConfig,
        NeutralWallLossConfig=wall_loss.NeutralWallLossConfig,
        ResamplingBuffers=exhaustion.ResamplingBuffers,
        NucleationConfig=nucleation.NucleationConfig,
        NucleationScratchBuffers=nucleation.NucleationScratchBuffers,
        NucleationFinalizedDemandBuffers=nucleation.NucleationFinalizedDemandBuffers,
        NucleationDiagnosticBuffers=nucleation.NucleationDiagnosticBuffers,
        NucleationExhaustionBuffers=nucleation.NucleationExhaustionBuffers,
        NucleationExhaustionControls=nucleation.NucleationExhaustionControls,
    )


def _allocate_sidecars(
    runtime: SimpleNamespace,
    device: str,
    dimensions: tuple[int, int, int],
    molar_mass: np.ndarray,
) -> SimpleNamespace:
    """Allocate all caller-owned direct-process sidecars once.

    Args:
        runtime: Lazily loaded Warp, helper, and concrete-record namespace.
        device: Active Warp device for every sidecar.
        dimensions: Fixed ``(B, N, S)`` process dimensions.
        molar_mass: Ordered CPU molar masses used by thermodynamics metadata.

    Returns:
        Namespace holding every sidecar and immutable direct-step metadata.
    """
    boxes, particles, species = dimensions
    wp = runtime.wp

    def f64(shape: tuple[int, ...]) -> Any:
        """Allocate a float64 sidecar on the active device."""
        return wp.zeros(shape, dtype=wp.float64, device=device)

    def i32(shape: tuple[int, ...]) -> Any:
        """Allocate an int32 sidecar on the active device."""
        return wp.zeros(shape, dtype=wp.int32, device=device)

    transfer_shape = (boxes, particles, species)
    scratch = runtime.CondensationScratchBuffers(
        dynamic_viscosity=f64((boxes,)),
        mean_free_path=f64((boxes,)),
        positive_mass_transfer_demand=f64((boxes, species)),
        negative_mass_transfer_release=f64((boxes, species)),
        positive_mass_transfer_scale=f64((boxes, species)),
    )
    resampling = runtime.ResamplingBuffers(
        i32((boxes,)),
        i32((boxes,)),
        i32((boxes, particles)),
        i32((boxes, particles)),
        i32((boxes, particles)),
        f64(transfer_shape),
        f64((boxes, particles)),
        f64((boxes, particles)),
        f64((boxes, particles)),
        f64((boxes,)),
        f64((boxes,)),
        f64((boxes,)),
        f64((boxes,)),
        i32((boxes,)),
    )
    exhaustion = runtime.NucleationExhaustionBuffers(
        resampling,
        f64((boxes,)),
        f64((boxes,)),
        wp.ones((boxes,), dtype=wp.float64, device=device),
        wp.ones((boxes,), dtype=wp.float64, device=device),
        wp.full((boxes,), 1.0e-12, dtype=wp.float64, device=device),
        f64((boxes,)),
        i32((boxes,)),
        i32((boxes,)),
        i32((boxes,)),
        i32((boxes,)),
        i32((boxes, particles)),
    )
    return SimpleNamespace(
        mass_transfer=f64(transfer_shape),
        condensation_scratch=scratch,
        thermodynamics=runtime.ThermodynamicsConfig(
            modes=i32((species,)),
            parameters=wp.array(
                np.column_stack(
                    (np.full(species, 1.0e-12), np.zeros((species, 3)))
                ),
                dtype=wp.float64,
                device=device,
            ),
            molar_mass_reference=wp.array(
                molar_mass, dtype=wp.float64, device=device
            ),
        ),
        collision_pairs=wp.full(
            (boxes, particles, 2), -1, dtype=wp.int32, device=device
        ),
        n_collisions=i32((boxes,)),
        coagulation_rng=wp.zeros((boxes,), dtype=wp.uint32, device=device),
        wall_rng=wp.zeros((boxes,), dtype=wp.uint32, device=device),
        nucleation_scratch=runtime.NucleationScratchBuffers(
            f64((boxes,)), f64((boxes,)), f64((boxes,))
        ),
        finalized_demand=runtime.NucleationFinalizedDemandBuffers(
            i32((boxes,)), f64((boxes,)), f64((boxes, species))
        ),
        diagnostics=runtime.NucleationDiagnosticBuffers(
            i32((boxes,)),
            wp.full((boxes, particles), -1, dtype=wp.int32, device=device),
            wp.full((boxes, particles), -1, dtype=wp.int32, device=device),
            i32((boxes,)),
            i32((boxes,)),
        ),
        exhaustion=exhaustion,
    )


def _output_prefix(
    particle_data: ParticleData,
    gas_data: GasData,
    environment_data: EnvironmentData,
) -> list[str]:
    """Build stable, address-free documentation output for the fixture."""
    return [
        "Canonical path: docs/Examples/gpu_complete_process_sequence.py",
        f"CPU shapes: particles={particle_data.masses.shape}, "
        f"gas={gas_data.concentration.shape}, "
        f"environment={environment_data.temperature.shape}",
        "Order: condensation -> coagulation -> dilution -> wall loss -> nucleation.",
        "Sidecars and RNG state are caller-owned and remain device-resident.",
    ]


def run_example(device: str = "cpu") -> ExampleRun:
    """Run the five direct boundaries with one setup and checkpoint transfer.

    A zero nucleation time demonstrates its direct boundary only; it is not a
    calibrated process schedule. Errors deliberately propagate without CPU
    fallback or an intermediate checkpoint.

    Args:
        device: Warp device for the explicit direct path; defaults to Warp CPU.

    Returns:
        Metadata-only disabled result, or final CPU data and owned sidecars.
    """
    particle_data, gas_data, environment_data, gas_names = _build_cpu_state()
    output = _output_prefix(particle_data, gas_data, environment_data)
    if not _warp_enabled():
        output.append("Warp is unavailable or disabled; no kernel ran.")
        return ExampleRun(output=output)
    runtime = _load_enabled_runtime()
    if runtime is None:
        output.append("Warp is unavailable or disabled; no kernel ran.")
        return ExampleRun(output=output)
    gpu_particles = runtime.gpu.to_warp_particle_data(
        particle_data, device=device
    )
    gpu_gas = runtime.gpu.to_warp_gas_data(gas_data, device=device)
    gpu_environment = runtime.gpu.to_warp_environment_data(
        environment_data, device=device
    )
    sidecars = _allocate_sidecars(
        runtime, device, particle_data.masses.shape, gas_data.molar_mass
    )
    wall_config = runtime.NeutralWallLossConfig(
        "spherical",
        0.01,
        chamber_radius=np.nextafter(0.0, 1.0),
        mode="charged",
        wall_potential=-12.0,
        wall_electric_field=3.0,
    )
    nucleation_config = runtime.NucleationConfig(
        rate_law="activation",
        coefficient=1.0e-12,
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1, 0),
        formation_diameter=1.0e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1.0e30,
        temperature_lower=200.0,
        temperature_upper=400.0,
    )
    mechanism_config = runtime.CoagulationMechanismConfig(
        mechanisms=("brownian",), distribution_type="particle_resolved"
    )
    returned_particles, transfer = runtime.condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        None,
        None,
        0.01,
        mass_transfer=sidecars.mass_transfer,
        environment=gpu_environment,
        thermodynamics=sidecars.thermodynamics,
        scratch_buffers=sidecars.condensation_scratch,
    )
    assert (
        returned_particles is gpu_particles
        and transfer is sidecars.mass_transfer
    )
    returned = runtime.coagulation_step_gpu(
        gpu_particles,
        None,
        None,
        1.0,
        max_collisions=sidecars.collision_pairs.shape[1],
        collision_pairs=sidecars.collision_pairs,
        n_collisions=sidecars.n_collisions,
        rng_states=sidecars.coagulation_rng,
        initialize_rng=True,
        environment=gpu_environment,
        mechanism_config=mechanism_config,
    )
    assert returned[0] is gpu_particles
    assert returned[1] is sidecars.collision_pairs
    assert returned[2] is sidecars.n_collisions
    returned_particles, returned_gas = runtime.dilution_step_gpu(
        gpu_particles, gpu_gas, 0.2, 0.1
    )
    assert returned_particles is gpu_particles and returned_gas is gpu_gas
    assert (
        runtime.wall_loss_step_gpu(
            gpu_particles,
            None,
            None,
            1.0,
            config=wall_config,
            rng_states=sidecars.wall_rng,
            initialize_rng=True,
            environment=gpu_environment,
        )
        is gpu_particles
    )
    returned_particles, returned_gas = runtime.nucleation_step_gpu(
        gpu_particles,
        gpu_gas,
        nucleation_config,
        0.0,
        scratch=sidecars.nucleation_scratch,
        finalized_demand=sidecars.finalized_demand,
        diagnostics=sidecars.diagnostics,
        exhaustion_controls=runtime.NucleationExhaustionControls(True, True),
        exhaustion_buffers=sidecars.exhaustion,
        environment=gpu_environment,
    )
    assert returned_particles is gpu_particles and returned_gas is gpu_gas
    runtime.wp.synchronize()
    restored_particles = runtime.gpu.from_warp_particle_data(
        gpu_particles, sync=False
    )
    restored_gas = runtime.gpu.from_warp_gas_data(
        gpu_gas, name=gas_names, sync=False
    )
    restored_environment = runtime.gpu.from_warp_environment_data(
        gpu_environment, sync=False
    )
    output.extend(
        [
            f"Explicit setup transfer and one final checkpoint: device={device}.",
            "Warp CPU is the default; CUDA is optional.",
            "No scheduler, backend selection, resident loop, Runnable, or CPU fallback.",
        ]
    )
    return ExampleRun(
        output=output,
        particle_data=restored_particles,
        gas_data=restored_gas,
        environment_data=restored_environment,
        mass_transfer=sidecars.mass_transfer,
        collision_pairs=sidecars.collision_pairs,
        n_collisions=sidecars.n_collisions,
        coagulation_rng=sidecars.coagulation_rng,
        wall_rng=sidecars.wall_rng,
        nucleation_diagnostics=sidecars.diagnostics,
    )


def main() -> None:
    """Run the example and print output only after a successful result."""
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
