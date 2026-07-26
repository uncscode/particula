"""Run five direct Warp steps with explicit setup and checkpoint transfers.

The enabled path converts CPU particle, gas, and environment containers once,
then calls condensation, coagulation, dilution, wall loss, and nucleation in
that order. It synchronizes once and restores each CPU container once after
all five calls succeed. Device selection, conversion, sidecars, RNG state,
synchronization, and restoration remain caller-owned.

Warp imports are lazy. A forced or naturally unavailable Warp runtime returns
deterministic no-kernel metadata without allocating, converting, synchronizing,
restoring, or selecting a CPU substitute. Direct-boundary errors propagate;
this example is not a scheduler, backend selector, resident loop, high-level
Runnable, or CPU fallback.
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
    """Store deterministic output and optional CPU and caller-owned GPU state.

    Every optional field is ``None`` when Warp is unavailable or disabled.

    Attributes:
        output: Deterministic, address-free status lines.
        particle_data: Particle data restored after the final synchronization.
        gas_data: Gas data restored with its original ordered names.
        environment_data: Environment data restored after the final
            synchronization.
        mass_transfer: Caller-owned condensation transfer sidecar.
        collision_pairs: Caller-owned coagulation collision-pair sidecar.
        n_collisions: Caller-owned coagulation count sidecar.
        coagulation_rng: Persistent caller-owned coagulation RNG sidecar.
        wall_rng: Persistent caller-owned wall-loss RNG sidecar.
        nucleation_diagnostics: Caller-owned nucleation diagnostic sidecars.
        dilution_particles: Resident particle container returned by dilution.
        dilution_gas: Resident gas container returned by dilution.
        wall_particles: Resident particle container returned by wall loss.
        nucleation_particles: Resident particle container returned by
            nucleation.
        nucleation_gas: Resident gas container returned by nucleation.
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
    dilution_particles: Any | None = None
    dilution_gas: Any | None = None
    wall_particles: Any | None = None
    nucleation_particles: Any | None = None
    nucleation_gas: Any | None = None


def _build_cpu_state() -> tuple[
    ParticleData, GasData, EnvironmentData, list[str]
]:
    """Create a deterministic one-box CPU fixture for the direct process path.

    The particle storage has two active and two exactly-free fixed slots. Gas
    names are copied before conversion because Warp gas data does not own them.

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
    """Check whether Warp can be probed without loading GPU helper modules.

    The force-disable environment variable short-circuits before the probe.
    An unavailable Warp import returns ``False`` without conversion, allocation,
    synchronization, restoration, or a CPU fallback.

    Returns:
        ``True`` when the Warp module imports; otherwise, ``False``.
    """
    if os.getenv(_FORCE_NO_WARP_ENV) == "1":
        return False
    try:
        importlib.import_module("warp")
    except ImportError:
        return False
    return True


def _load_enabled_runtime() -> SimpleNamespace | None:
    """Lazily load enabled-only direct boundaries and their concrete records.

    This loader is reached only after the standalone Warp probe succeeds. It
    binds the five public direct calls and keeps concrete configuration and
    sidecar records at their owning modules.

    Returns:
        The enabled Warp runtime namespace, or ``None`` when the GPU package
        reports that Warp is unavailable.
    """
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
        to_warp_particle_data=gpu.to_warp_particle_data,
        to_warp_gas_data=gpu.to_warp_gas_data,
        to_warp_environment_data=gpu.to_warp_environment_data,
        from_warp_particle_data=gpu.from_warp_particle_data,
        from_warp_gas_data=gpu.from_warp_gas_data,
        from_warp_environment_data=gpu.from_warp_environment_data,
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
        NucleationFinalizedDemandBuffers=(
            nucleation.NucleationFinalizedDemandBuffers
        ),
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
    """Allocate all caller-owned direct-process sidecars exactly once.

    Args:
        runtime: Lazily loaded Warp, helper, and concrete-record namespace.
        device: Active Warp device for every sidecar.
        dimensions: Fixed ``(B, N, S)`` process dimensions.
        molar_mass: Ordered CPU molar masses used by thermodynamics metadata.

    Returns:
        Namespace holding every sidecar and immutable direct-step metadata.
        Every Warp array uses ``device`` and the fixed process dimensions.
    """
    boxes, particles, species = dimensions
    wp = runtime.wp

    def f64(shape: tuple[int, ...]) -> Any:
        return wp.zeros(shape, dtype=wp.float64, device=device)

    def i32(shape: tuple[int, ...]) -> Any:
        return wp.zeros(shape, dtype=wp.int32, device=device)

    def u32(shape: tuple[int, ...]) -> Any:
        return wp.zeros(shape, dtype=wp.uint32, device=device)

    transfer_shape = (boxes, particles, species)
    resampling = runtime.ResamplingBuffers(
        retained_counts=i32((boxes,)),
        released_counts=i32((boxes,)),
        retained_indices=i32((boxes, particles)),
        released_indices=i32((boxes, particles)),
        sorted_indices=i32((boxes, particles)),
        replacement_masses=f64(transfer_shape),
        replacement_concentration=f64((boxes, particles)),
        replacement_charge=f64((boxes, particles)),
        source_radii=f64((boxes, particles)),
        radius_cubed_relative_error=f64((boxes,)),
        mean_radius_relative_error=f64((boxes,)),
        surface_relative_error=f64((boxes,)),
        diversity_absolute_error=f64((boxes,)),
        planning_status=i32((boxes,)),
    )
    return SimpleNamespace(
        mass_transfer=f64(transfer_shape),
        condensation_scratch=runtime.CondensationScratchBuffers(
            work_mass_transfer=None,
            total_mass_transfer=None,
            dynamic_viscosity=f64((boxes,)),
            mean_free_path=f64((boxes,)),
            positive_mass_transfer_demand=f64((boxes, species)),
            negative_mass_transfer_release=f64((boxes, species)),
            positive_mass_transfer_scale=f64((boxes, species)),
        ),
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
        coagulation_rng=u32((boxes,)),
        wall_rng=u32((boxes,)),
        resampling=resampling,
        nucleation_scratch=runtime.NucleationScratchBuffers(
            precursor_number_concentration=f64((boxes,)),
            potential_rate=f64((boxes,)),
            potential_demand=f64((boxes,)),
        ),
        finalized_demand=runtime.NucleationFinalizedDemandBuffers(
            accepted_counts=i32((boxes,)),
            accepted_demand=f64((boxes,)),
            precursor_mass_change=f64((boxes, species)),
        ),
        diagnostics=runtime.NucleationDiagnosticBuffers(
            gate_codes=i32((boxes,)),
            selected_slot_indices=wp.full(
                (boxes, particles), -1, dtype=wp.int32, device=device
            ),
            free_slot_indices=wp.full(
                (boxes, particles), -1, dtype=wp.int32, device=device
            ),
            active_slot_counts=i32((boxes,)),
            free_slot_counts=i32((boxes,)),
        ),
        exhaustion=runtime.NucleationExhaustionBuffers(
            resampling_buffers=resampling,
            demand_workspace=f64((boxes,)),
            final_demand=f64((boxes,)),
            requested_scale=wp.ones((boxes,), dtype=wp.float64, device=device),
            minimum_scale=wp.ones((boxes,), dtype=wp.float64, device=device),
            minimum_volume=wp.full(
                (boxes,), 1.0e-12, dtype=wp.float64, device=device
            ),
            resolved_scale=f64((boxes,)),
            resampling_releasable_counts=i32((boxes,)),
            required_release_counts=i32((boxes,)),
            scaling_required=i32((boxes,)),
            final_counts=i32((boxes,)),
            final_selected_slot_indices=i32((boxes, particles)),
        ),
    )


def _output_prefix(
    particle_data: ParticleData,
    gas_data: GasData,
    environment_data: EnvironmentData,
) -> list[str]:
    """Build stable, address-free contract output for the CPU fixture.

    Args:
        particle_data: CPU particle fixture used to report capacity shape.
        gas_data: CPU gas fixture used to report concentration shape.
        environment_data: CPU environment fixture used to report box shape.

    Returns:
        User-facing lines describing ordering, ownership, and exclusions.
    """
    return [
        "Canonical path: docs/Examples/gpu_complete_process_sequence.py",
        (
            "CPU fixture: "
            f"particles={particle_data.masses.shape}, "
            f"gas={gas_data.concentration.shape}, "
            f"environment={environment_data.temperature.shape}"
        ),
        (
            "Process order: condensation -> coagulation -> dilution -> "
            "wall loss -> nucleation."
        ),
        (
            "Ownership: conversions, sidecars, RNG state, synchronization, "
            "and the final restore stay caller-owned."
        ),
        ("Runtime: Warp CPU is the default when installed; CUDA is optional."),
        (
            "Exclusions: no scheduler, backend selection, resident loop, "
            "Runnable, or CPU fallback."
        ),
    ]


def run_example(device: str = "cpu") -> ExampleRun:
    """Run five direct boundaries with one setup transfer and checkpoint.

    The enabled path makes exactly five direct calls in this order:
    condensation, coagulation, dilution, wall loss, and nucleation. It retains
    the same resident containers and caller-owned sidecars across those calls,
    then synchronizes and restores only once. A zero nucleation time
    demonstrates its direct boundary only; it is not a calibrated schedule.
    Errors deliberately propagate without fallback or an intermediate restore.

    Args:
        device: Warp device for the explicit direct path; defaults to Warp CPU.

    Returns:
        Deterministic no-kernel metadata when Warp is disabled or unavailable;
        otherwise, final restored CPU data and caller-owned sidecars.

    Raises:
        ImportError: If enabled-only runtime loading cannot import a dependency.
        ValueError: If a direct boundary rejects the supplied process state.
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
        geometry="spherical",
        wall_eddy_diffusivity=0.01,
        chamber_radius=0.5,
        distribution_type="particle_resolved",
        mode="charged",
        wall_potential=0.05,
        wall_electric_field=0.0,
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
        mechanisms=("brownian",),
        distribution_type="particle_resolved",
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
    dilution_particles, dilution_gas = runtime.dilution_step_gpu(
        gpu_particles, gpu_gas, 0.2, 0.1
    )
    assert dilution_particles is gpu_particles
    assert dilution_gas is gpu_gas
    wall_particles = runtime.wall_loss_step_gpu(
        gpu_particles,
        None,
        None,
        1.0,
        config=wall_config,
        rng_states=sidecars.wall_rng,
        initialize_rng=True,
        environment=gpu_environment,
    )
    assert wall_particles is gpu_particles
    nucleation_particles, nucleation_gas = runtime.nucleation_step_gpu(
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
    assert nucleation_particles is gpu_particles
    assert nucleation_gas is gpu_gas
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
            f"Enabled path: device={device}, one setup transfer, one sync, "
            "and one final checkpoint.",
            "Direct outputs remain caller-owned: condensation transfer, "
            "coagulation buffers, dilution containers, wall particles, "
            "nucleation containers, and diagnostic/RNG sidecars.",
        ]
    )
    return ExampleRun(
        output=output,
        particle_data=restored_particles,
        gas_data=restored_gas,
        environment_data=restored_environment,
        mass_transfer=transfer,
        collision_pairs=returned[1],
        n_collisions=returned[2],
        coagulation_rng=sidecars.coagulation_rng,
        wall_rng=sidecars.wall_rng,
        nucleation_diagnostics=sidecars.diagnostics,
        dilution_particles=dilution_particles,
        dilution_gas=dilution_gas,
        wall_particles=wall_particles,
        nucleation_particles=nucleation_particles,
        nucleation_gas=nucleation_gas,
    )


def main() -> None:
    """Run the example and print only its completed-result output.

    Exceptions are intentionally not caught, so an enabled-path failure cannot
    print a misleading success message or select a CPU fallback.
    """
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
