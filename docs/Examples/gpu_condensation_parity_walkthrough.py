"""Compare an independent NumPy oracle with direct Warp condensation.

This direct-kernel-only walkthrough defaults to Warp's CPU device and makes
all CPU↔Warp transfers explicit. It first completes an independent four
substep NumPy calculation, then optionally runs the low-level Warp route.  If
Warp is unavailable, or ``PARTICULA_EXAMPLE_FORCE_NO_WARP=1``, it reports the
completed oracle and does not import a kernel or allocate device state.

Run ``python docs/Examples/gpu_condensation_parity_walkthrough.py`` or
``pytest particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py -q
-Werror``. Warp outputs are mutable caller-owned state. ``energy_transfer``
is caller-owned ``(n_boxes, n_species)`` storage, mutated in place with signed
joules, and is not a third returned value. A failed kernel call
may have partially mutated that detached state; discard it and build fresh
sources from the immutable fixture before retrying.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
    get_mass_transfer_rate_latent_heat,
)
from particula.gas import GasData
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.gas.properties.pressure_function import get_partial_pressure
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)
from particula.gpu import (
    WARP_AVAILABLE,
    from_warp_gas_data,
    from_warp_particle_data,
    to_warp_gas_data,
    to_warp_particle_data,
)
from particula.particles import ParticleData
from particula.particles.properties.aerodynamic_mobility_module import (
    get_aerodynamic_mobility,
)
from particula.particles.properties.diffusion_coefficient import (
    get_diffusion_coefficient,
)
from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius,
    get_kelvin_term,
)
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number,
)
from particula.particles.properties.slip_correction_module import (
    get_cunningham_slip_correction,
)
from particula.particles.properties.vapor_correction_module import (
    get_vapor_transition_correction,
)
from particula.util import constants

_FORCE_NO_WARP_ENV = "PARTICULA_EXAMPLE_FORCE_NO_WARP"
_TIME_STEP = 0.08


def _readonly(values: np.ndarray) -> np.ndarray:
    """Return an immutable float64 template."""
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _readonly_bool(values: np.ndarray) -> np.ndarray:
    """Return an immutable Boolean template."""
    result = np.asarray(values, dtype=bool)
    result.setflags(write=False)
    return result


def _warp_array(wp: Any, values: np.ndarray, dtype: Any, device: str) -> Any:
    """Create an explicit same-device Warp array from detached host values."""
    return wp.array(values, dtype=dtype, device=device)


@dataclass(frozen=True)
class ParityFixture:
    """Store immutable two-box/two-particle/two-species fp64 templates.

    Particle mass arrays use ``(box, particle, species)`` and gas arrays use
    ``(box, species)``.  Species order is shared by masses, gas names, and all
    species sidecars.
    """

    temperature: np.ndarray
    pressure: np.ndarray
    masses: np.ndarray
    concentration: np.ndarray
    charge: np.ndarray
    density: np.ndarray
    volume: np.ndarray
    names: tuple[str, str]
    molar_mass: np.ndarray
    gas_concentration: np.ndarray
    partitioning: np.ndarray
    thermodynamic_modes: np.ndarray
    thermodynamic_parameters: np.ndarray
    surface_tension: np.ndarray
    mass_accommodation: np.ndarray
    diffusion_coefficient_vapor: np.ndarray
    latent_heat: np.ndarray


def build_fixture() -> ParityFixture:
    """Build immutable templates with simultaneous uptake and evaporation.

    Returns:
        Immutable two-box, two-particle, two-species fp64 example fixture.
    """
    modes = np.array([0, 0], dtype=np.int32)
    modes.setflags(write=False)
    return ParityFixture(
        temperature=_readonly(np.array([298.15, 305.15])),
        pressure=_readonly(np.array([101325.0, 98000.0])),
        masses=_readonly(
            np.array(
                [
                    [[1.0e-18, 2.0e-18], [2.0e-18, 1.0e-18]],
                    [[1.5e-18, 1.0e-18], [2.5e-18, 2.0e-18]],
                ]
            )
        ),
        concentration=_readonly(np.array([[2.0e6, 1.0e6], [1.5e6, 2.5e6]])),
        charge=_readonly(np.zeros((2, 2))),
        density=_readonly(np.array([1000.0, 1100.0])),
        volume=_readonly(np.array([1.0e-6, 1.2e-6])),
        names=("Water", "Organic"),
        molar_mass=_readonly(np.array([0.018015, 0.150])),
        gas_concentration=_readonly(
            np.array([[2.0e-6, 5.0e-8], [1.5e-6, 8.0e-8]])
        ),
        partitioning=_readonly_bool(np.array([True, True])),
        thermodynamic_modes=modes,
        thermodynamic_parameters=_readonly(
            np.array([[0.01, 0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.0]])
        ),
        surface_tension=_readonly(np.array([0.072, 0.035])),
        mass_accommodation=_readonly(np.array([1.0, 0.8])),
        diffusion_coefficient_vapor=_readonly(np.array([2.1e-5, 8.0e-6])),
        latent_heat=_readonly(np.array([2.26e6, 4.0e5])),
    )


@dataclass
class OracleInput:
    """Detached CPU state for the independent oracle."""

    particles: ParticleData
    gas: GasData
    fixture: ParityFixture


@dataclass
class WarpSource:
    """Detached CPU state converted only after optional Warp enablement."""

    particles: ParticleData
    gas: GasData
    vapor_pressure: np.ndarray
    fixture: ParityFixture


@dataclass
class OracleResult:
    """Record independent final state and direct-kernel-compatible diagnostics."""

    masses: np.ndarray
    gas_concentration: np.ndarray
    total_mass_transfer: np.ndarray
    raw_proposal: np.ndarray
    energy_transfer: np.ndarray
    substeps: tuple["OracleSubstep", ...]


@dataclass(frozen=True)
class OracleSubstep:
    """Record one independent fixed substep for walkthrough diagnostics."""

    time_step: float
    gas_concentration_before: np.ndarray
    raw_proposal: np.ndarray
    finalized_transfer: np.ndarray


@dataclass(frozen=True)
class AcceptanceResult:
    """Record one categorized acceptance outcome and diagnostic.

    The categories independently report physics parity, inventory
    conservation, and signed energy bookkeeping.
    """

    category: Literal["physics", "conservation", "energy"]
    status: Literal["passed", "failed", "unavailable"]
    diagnostic: str


@dataclass
class ExampleRun:
    """Return the oracle, optional Warp observations, and acceptance results.

    ``acceptance`` always contains physics, conservation, and energy results.
    They are marked unavailable when the optional Warp route does not run.
    """

    output: list[str]
    oracle: OracleResult
    particle_data: ParticleData | None = None
    gas_data: GasData | None = None
    raw_proposal: np.ndarray | None = None
    total_mass_transfer: np.ndarray | None = None
    energy_transfer: np.ndarray | None = None
    vapor_pressure: np.ndarray | None = None
    acceptance: tuple[AcceptanceResult, ...] = ()
    scratch_buffers: Any | None = None


def _validate_float_fields(
    float_shapes: dict[str, tuple[np.ndarray, tuple[int, ...]]],
) -> None:
    """Validate floating-point array shapes and values."""
    for name, (values, shape) in float_shapes.items():
        if values.shape != shape or values.dtype != np.float64:
            message = f"{name} must be float64 with shape {shape}."
            raise ValueError(message)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")


def _validate_signed_fields(
    fixture: ParityFixture,
    field_names: tuple[str, ...],
    strictly_positive: bool,
    description: str,
) -> None:
    """Validate a shared lower bound for named fixture fields."""
    for name in field_names:
        values = getattr(fixture, name)
        invalid_values = values <= 0.0 if strictly_positive else values < 0.0
        if np.any(invalid_values):
            raise ValueError(f"{name} must contain only {description} values.")


def _validate_fixture(fixture: ParityFixture) -> None:
    """Validate canonical example dimensions before detached construction."""
    float_shapes = {
        "temperature": (fixture.temperature, (2,)),
        "pressure": (fixture.pressure, (2,)),
        "masses": (fixture.masses, (2, 2, 2)),
        "concentration": (fixture.concentration, (2, 2)),
        "charge": (fixture.charge, (2, 2)),
        "density": (fixture.density, (2,)),
        "volume": (fixture.volume, (2,)),
        "molar_mass": (fixture.molar_mass, (2,)),
        "gas_concentration": (fixture.gas_concentration, (2, 2)),
        "thermodynamic_parameters": (
            fixture.thermodynamic_parameters,
            (2, 4),
        ),
        "surface_tension": (fixture.surface_tension, (2,)),
        "mass_accommodation": (fixture.mass_accommodation, (2,)),
        "diffusion_coefficient_vapor": (
            fixture.diffusion_coefficient_vapor,
            (2,),
        ),
        "latent_heat": (fixture.latent_heat, (2,)),
    }
    _validate_float_fields(float_shapes)
    positive_fields = (
        "temperature",
        "pressure",
        "density",
        "volume",
        "molar_mass",
    )
    nonnegative_fields = (
        "masses",
        "concentration",
        "gas_concentration",
        "surface_tension",
        "mass_accommodation",
        "diffusion_coefficient_vapor",
        "latent_heat",
    )
    _validate_signed_fields(fixture, positive_fields, True, "positive")
    _validate_signed_fields(fixture, nonnegative_fields, False, "nonnegative")
    if fixture.partitioning.shape != (2,) or fixture.partitioning.dtype != bool:
        raise ValueError("partitioning must be Boolean with shape (2,).")
    if (
        fixture.thermodynamic_modes.shape != (2,)
        or fixture.thermodynamic_modes.dtype != np.int32
    ):
        raise ValueError("thermodynamic_modes must be int32 with shape (2,).")
    if len(fixture.names) != 2:
        raise ValueError("names must contain the two species-aligned labels.")


def _particle_data(fixture: ParityFixture) -> ParticleData:
    """Build non-aliasing particle data with species-aligned mass columns."""
    return ParticleData(
        masses=fixture.masses.copy(),
        concentration=fixture.concentration.copy(),
        charge=fixture.charge.copy(),
        density=fixture.density.copy(),
        volume=fixture.volume.copy(),
    )


def build_oracle_input(fixture: ParityFixture | None = None) -> OracleInput:
    """Build detached oracle inputs without aliasing immutable templates.

    Args:
        fixture: Optional validated immutable template. A canonical template is
            built when omitted.

    Returns:
        Detached CPU particle and gas state for the independent oracle.
    """
    fixture = build_fixture() if fixture is None else fixture
    _validate_fixture(fixture)
    return OracleInput(
        particles=_particle_data(fixture),
        gas=GasData(
            name=list(fixture.names),
            molar_mass=fixture.molar_mass.copy(),
            concentration=fixture.gas_concentration.copy(),
            partitioning=fixture.partitioning.copy(),
        ),
        fixture=fixture,
    )


def build_warp_source(fixture: ParityFixture | None = None) -> WarpSource:
    """Build separately detached conversion inputs and derived pressure storage.

    Args:
        fixture: Optional validated immutable template. A canonical template is
            built when omitted.

    Returns:
        Detached CPU conversion source and zeroed derived vapor-pressure storage.
    """
    fixture = build_fixture() if fixture is None else fixture
    _validate_fixture(fixture)
    return WarpSource(
        particles=_particle_data(fixture),
        gas=GasData(
            name=list(fixture.names),
            molar_mass=fixture.molar_mass.copy(),
            concentration=fixture.gas_concentration.copy(),
            partitioning=fixture.partitioning.copy(),
        ),
        vapor_pressure=np.zeros((2, 2), dtype=np.float64),
        fixture=fixture,
    )


def _proposal(state: OracleInput) -> np.ndarray:  # noqa: C901
    """Compute one predecessor-state raw transfer proposal using public helpers."""
    fixture = state.fixture
    masses = state.particles.masses
    proposal = np.zeros_like(masses)
    vapor_pressure = np.broadcast_to(
        fixture.thermodynamic_parameters[:, 0], (2, 2)
    )
    for box in range(2):
        temperature = fixture.temperature[box]
        viscosity = get_dynamic_viscosity(
            temperature,
            constants.REF_VISCOSITY_AIR_STP,
            constants.REF_TEMPERATURE_STP,
        )
        mean_free_path = get_molecule_mean_free_path(
            constants.MOLECULAR_WEIGHT_AIR,
            temperature,
            fixture.pressure[box],
            viscosity,
        )
        for particle in range(2):
            if state.particles.concentration[box, particle] == 0.0:
                continue
            particle_masses = masses[box, particle]
            total_volume = np.sum(particle_masses / fixture.density)
            if total_volume <= 0.0:
                continue
            radius = (3.0 * total_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
            density = np.sum(particle_masses) / total_volume
            knudsen = get_knudsen_number(mean_free_path, radius)
            slip = get_cunningham_slip_correction(knudsen)
            mobility = get_aerodynamic_mobility(radius, slip, viscosity)
            particle_diffusion = get_diffusion_coefficient(
                temperature, mobility, constants.BOLTZMANN_CONSTANT
            )
            for species in range(2):
                transition = get_vapor_transition_correction(
                    knudsen, fixture.mass_accommodation[species]
                )
                vapor_diffusion = fixture.diffusion_coefficient_vapor[species]
                if vapor_diffusion <= 0.0:
                    vapor_diffusion = particle_diffusion
                transport = get_first_order_mass_transport_k(
                    radius, transition, vapor_diffusion
                )
                kelvin_radius = get_kelvin_radius(
                    fixture.surface_tension[species],
                    density,
                    state.gas.molar_mass[species],
                    temperature,
                )
                surface_pressure = vapor_pressure[
                    box, species
                ] * get_kelvin_term(radius, kelvin_radius)
                delta_pressure = (
                    get_partial_pressure(
                        state.gas.concentration[box, species],
                        state.gas.molar_mass[species],
                        temperature,
                    )
                    - surface_pressure
                )
                proposal[box, particle, species] = (
                    get_mass_transfer_rate_latent_heat(
                        delta_pressure,
                        transport,
                        temperature,
                        state.gas.molar_mass[species],
                        fixture.latent_heat[species],
                        get_thermal_conductivity(temperature),
                        surface_pressure,
                        vapor_diffusion,
                    )
                    * (_TIME_STEP / 4.0)
                )
    proposal[:, :, ~fixture.partitioning] = 0.0
    return proposal


def _finalize_transfer(state: OracleInput, proposal: np.ndarray) -> np.ndarray:
    """Apply P2 bounds and ordered concentration-weighted inventory limiting."""
    bounded = np.maximum(proposal, -state.particles.masses)
    applied = bounded.copy()
    for box in range(2):
        for species in range(2):
            demand = 0.0
            release = 0.0
            for particle in range(2):
                weighted = (
                    bounded[box, particle, species]
                    * state.particles.concentration[box, particle]
                )
                if weighted > 0.0:
                    demand += weighted
                else:
                    release -= weighted
            scale = (
                np.clip(
                    (state.gas.concentration[box, species] + release) / demand,
                    0.0,
                    1.0,
                )
                if demand > 0.0
                else 1.0
            )
            applied[box, :, species] = np.where(
                bounded[box, :, species] > 0.0,
                bounded[box, :, species] * scale,
                bounded[box, :, species],
            )
    return applied


def run_oracle(source: OracleInput | None = None) -> OracleResult:
    """Run exactly four independent fixed substeps before any Warp action.

    Args:
        source: Optional detached CPU state. A canonical detached source is
            built when omitted and is mutated only within this call.

    Returns:
        Final independent state, coupled transfer diagnostics, signed-joule
        energy diagnostic, and immutable per-substep observations.
    """
    state = build_oracle_input() if source is None else source
    total = np.zeros_like(state.particles.masses)
    raw_proposal = np.zeros_like(total)
    substeps: list[OracleSubstep] = []
    for _ in range(4):
        gas_before = state.gas.concentration.copy()
        raw_proposal = _proposal(state)
        applied = _finalize_transfer(state, raw_proposal)
        substeps.append(
            OracleSubstep(
                time_step=_TIME_STEP / 4.0,
                gas_concentration_before=gas_before,
                raw_proposal=raw_proposal.copy(),
                finalized_transfer=applied.copy(),
            )
        )
        state.particles.masses += applied
        state.gas.concentration -= np.sum(
            applied * state.particles.concentration[:, :, None], axis=1
        )
        np.maximum(state.gas.concentration, 0.0, out=state.gas.concentration)
        total += applied
    energy = np.sum(total, axis=1) * state.fixture.latent_heat[None, :]
    return OracleResult(
        masses=state.particles.masses.copy(),
        gas_concentration=state.gas.concentration.copy(),
        total_mass_transfer=total,
        raw_proposal=raw_proposal,
        energy_transfer=energy,
        substeps=tuple(substeps),
    )


def _acceptance_result(
    category: Literal["physics", "conservation", "energy"],
    evaluate: Any,
) -> AcceptanceResult:
    """Evaluate one criterion and retain any assertion diagnostic.

    Args:
        category: Label for the acceptance criterion.
        evaluate: Zero-argument assertion-based criterion evaluator.

    Returns:
        Passed or failed result with a diagnostic string.
    """
    try:
        evaluate()
    except AssertionError as error:
        return AcceptanceResult(category, "failed", str(error))
    return AcceptanceResult(category, "passed", "criteria satisfied")


def evaluate_acceptance(
    fixture: ParityFixture,
    oracle: OracleResult,
    observed_masses: np.ndarray,
    observed_gas_concentration: np.ndarray,
    observed_total_mass_transfer: np.ndarray,
    observed_energy_transfer: np.ndarray,
    observed_vapor_pressure: np.ndarray,
) -> tuple[AcceptanceResult, ...]:
    """Evaluate independent physics, conservation, and energy criteria.

    Every criterion is evaluated even when an earlier criterion fails.

    Args:
        fixture: Immutable initial state and physical parameters.
        oracle: Independent NumPy final state and transfer diagnostic.
        observed_masses: Synchronized Warp final particle masses in kg.
        observed_gas_concentration: Synchronized Warp final gas concentrations.
        observed_total_mass_transfer: Synchronized finalized particle transfers
            in kg.
        observed_energy_transfer: Synchronized signed energy sidecar in J.
        observed_vapor_pressure: Synchronized derived vapor pressure in Pa.

    Returns:
        Physics, conservation, and energy results in that order.
    """
    expected_vapor_pressure = np.broadcast_to(
        fixture.thermodynamic_parameters[:, 0], (2, 2)
    )

    def evaluate_physics() -> None:
        np.testing.assert_allclose(
            observed_masses, oracle.masses, rtol=2e-10, atol=1e-30
        )
        np.testing.assert_allclose(
            observed_gas_concentration,
            oracle.gas_concentration,
            rtol=2e-10,
            atol=1e-30,
        )
        np.testing.assert_allclose(
            observed_total_mass_transfer,
            oracle.total_mass_transfer,
            rtol=2e-10,
            atol=1e-30,
        )
        np.testing.assert_array_equal(
            observed_vapor_pressure, expected_vapor_pressure
        )

    def evaluate_conservation() -> None:
        particle_inventory_change = np.sum(
            (observed_masses - fixture.masses)
            * fixture.concentration[:, :, None],
            axis=1,
        )
        gas_change = observed_gas_concentration - fixture.gas_concentration
        drift = particle_inventory_change + gas_change
        conservation_scale = np.maximum(
            np.abs(fixture.gas_concentration),
            np.abs(observed_gas_concentration),
        )
        np.testing.assert_allclose(
            drift + conservation_scale,
            conservation_scale,
            rtol=1e-12,
            atol=1e-30,
        )

    def evaluate_energy() -> None:
        expected_energy = (
            np.sum(observed_total_mass_transfer, axis=1)
            * fixture.latent_heat[None, :]
        )
        np.testing.assert_allclose(
            observed_energy_transfer, expected_energy, rtol=1e-12, atol=1e-18
        )

    return (
        _acceptance_result("physics", evaluate_physics),
        _acceptance_result("conservation", evaluate_conservation),
        _acceptance_result("energy", evaluate_energy),
    )


def _format_acceptance(result: AcceptanceResult) -> str:
    """Format one machine-testable acceptance block for script output.

    Args:
        result: Categorized acceptance result to display.

    Returns:
        One labeled, human-readable output line.
    """
    return f"{result.category}: {result.status} - {result.diagnostic}"


def _warp_enabled() -> bool:
    """Return whether optional Warp execution is available and not disabled."""
    return WARP_AVAILABLE and os.getenv(_FORCE_NO_WARP_ENV) != "1"


def _load_gpu_runtime() -> tuple[Any, Any, Any, Any]:
    """Lazily import Warp and concrete direct-kernel-only runtime modules."""
    wp = importlib.import_module("warp")
    kernels = importlib.import_module("particula.gpu.kernels")
    condensation = importlib.import_module("particula.gpu.kernels.condensation")
    thermodynamics = importlib.import_module(
        "particula.gpu.kernels.thermodynamics"
    )
    return (
        wp,
        kernels.condensation_step_gpu,
        condensation.CondensationScratchBuffers,
        thermodynamics.ThermodynamicsConfig,
    )


def run_example(device: str = "cpu") -> ExampleRun:
    """Complete the oracle then optionally execute and synchronize Warp state.

    Enabled-route errors intentionally propagate.  The detached source and its
    sidecars must be discarded after failure because opaque kernel mutation is
    not atomic; the completed oracle remains an independent clean result.

    Args:
        device: Warp device for the optional direct-kernel route. Defaults to
            Warp's CPU device.

    Returns:
        Completed oracle and, when Warp is enabled, synchronized mutable
        caller-owned outputs. ``energy_transfer`` has shape
        ``(n_boxes, n_species)``, records signed joules in place, and is not a
        third value returned by the direct kernel step.
    """
    fixture = build_fixture()
    oracle = run_oracle(build_oracle_input(fixture))
    output = ["Independent NumPy oracle completed (four fixed substeps)."]
    if not _warp_enabled():
        output.append("oracle completed; no kernel ran")
        acceptance = tuple(
            AcceptanceResult(category, "unavailable", "no-Warp observations")
            for category in ("physics", "conservation", "energy")
        )
        output.extend(_format_acceptance(result) for result in acceptance)
        return ExampleRun(output=output, oracle=oracle, acceptance=acceptance)
    wp, step, scratch_type, thermodynamics_type = _load_gpu_runtime()
    source = build_warp_source(fixture)
    names = list(source.gas.name)
    particles = to_warp_particle_data(source.particles, device=device)
    gas = to_warp_gas_data(
        source.gas, device=device, vapor_pressure=source.vapor_pressure
    )
    transfer_shape, box_shape, gas_shape = (2, 2, 2), (2,), (2, 2)
    scratch = scratch_type(
        work_mass_transfer=wp.zeros(
            transfer_shape, dtype=wp.float64, device=device
        ),
        total_mass_transfer=wp.zeros(
            transfer_shape, dtype=wp.float64, device=device
        ),
        dynamic_viscosity=wp.zeros(box_shape, dtype=wp.float64, device=device),
        mean_free_path=wp.zeros(box_shape, dtype=wp.float64, device=device),
        positive_mass_transfer_demand=wp.zeros(
            gas_shape, dtype=wp.float64, device=device
        ),
        negative_mass_transfer_release=wp.zeros(
            gas_shape, dtype=wp.float64, device=device
        ),
        positive_mass_transfer_scale=wp.zeros(
            gas_shape, dtype=wp.float64, device=device
        ),
    )
    thermodynamics = thermodynamics_type(
        modes=_warp_array(wp, fixture.thermodynamic_modes, wp.int32, device),
        parameters=_warp_array(
            wp, fixture.thermodynamic_parameters, wp.float64, device
        ),
        molar_mass_reference=_warp_array(
            wp, fixture.molar_mass, wp.float64, device
        ),
    )
    energy = wp.zeros(gas_shape, dtype=wp.float64, device=device)
    _, total = step(
        particles,
        gas,
        temperature=_warp_array(wp, fixture.temperature, wp.float64, device),
        pressure=_warp_array(wp, fixture.pressure, wp.float64, device),
        time_step=_TIME_STEP,
        surface_tension=_warp_array(
            wp, fixture.surface_tension, wp.float64, device
        ),
        mass_accommodation=_warp_array(
            wp, fixture.mass_accommodation, wp.float64, device
        ),
        diffusion_coefficient_vapor=_warp_array(
            wp, fixture.diffusion_coefficient_vapor, wp.float64, device
        ),
        thermodynamics=thermodynamics,
        scratch_buffers=scratch,
        latent_heat=_warp_array(wp, fixture.latent_heat, wp.float64, device),
        energy_transfer=energy,
    )
    assert total is scratch.total_mass_transfer
    wp.synchronize_device(device)
    restored_particles = from_warp_particle_data(particles)
    wp.synchronize_device(device)
    restored_gas = from_warp_gas_data(gas, name=names)
    wp.synchronize_device(device)
    observed_vapor_pressure = gas.vapor_pressure.numpy().copy()
    wp.synchronize_device(device)
    raw = scratch.work_mass_transfer.numpy().copy()
    wp.synchronize_device(device)
    applied = total.numpy().copy()
    wp.synchronize_device(device)
    observed_energy = energy.numpy().copy()
    acceptance = evaluate_acceptance(
        fixture,
        oracle,
        restored_particles.masses,
        restored_gas.concentration,
        applied,
        observed_energy,
        observed_vapor_pressure,
    )
    output.append("Synchronized Warp direct-kernel observations completed.")
    output.extend(_format_acceptance(result) for result in acceptance)
    return ExampleRun(
        output=output,
        oracle=oracle,
        particle_data=restored_particles,
        gas_data=restored_gas,
        raw_proposal=raw,
        total_mass_transfer=applied,
        energy_transfer=observed_energy,
        vapor_pressure=observed_vapor_pressure,
        acceptance=acceptance,
        scratch_buffers=scratch,
    )


def main() -> None:
    """Run the walkthrough and print independent and optional Warp labels."""
    for line in run_example().output:
        print(line)


if __name__ == "__main__":
    main()
