"""Native CPU and resident-Warp condensation adapter integration evidence.

The CPU path deliberately uses the legacy aerosol model.  Warp rows retain
caller-owned data on their device and check the direct four-substep contract
without using the CPU runnable as an oracle.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

from particula.aerosol import Aerosol
from particula.dynamics import CondensationIsothermal, MassCondensation
from particula.dynamics.condensation.mass_transfer import (
    get_first_order_mass_transport_k,
    get_mass_transfer_rate,
    get_mass_transfer_rate_latent_heat,
)
from particula.execution import (
    CondensationActivityMode,
    CondensationConfiguration,
    CondensationExecutionMode,
    CondensationSurfaceMode,
    ExecutionResult,
)
from particula.execution.adapters.condensation import (
    CondensationExecutionConfig,
    CPUCondensationExecutionAdapter,
    CPUCondensationExecutionState,
    CPUCondensationState,
    WarpCondensationExecutionAdapter,
    WarpCondensationExecutionState,
    WarpCondensationState,
)
from particula.gas import AtmosphereBuilder, EnvironmentData, GasSpecies
from particula.gas.gas_data import GasData
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gas.properties.mean_free_path import get_molecule_mean_free_path
from particula.gas.properties.pressure_function import get_partial_pressure
from particula.gas.properties.thermal_conductivity import (
    get_thermal_conductivity,
)
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import (
    ActivityIdealMass,
    ParticleResolvedSpeciatedMass,
    ResolvedParticleMassRepresentationBuilder,
    SurfaceStrategyVolume,
)
from particula.particles.particle_data import ParticleData
from particula.particles.properties.kelvin_effect_module import (
    get_kelvin_radius,
    get_kelvin_term,
)
from particula.particles.properties.knudsen_number_module import (
    get_knudsen_number,
)
from particula.particles.properties.vapor_correction_module import (
    get_vapor_transition_correction,
)
from particula.util import constants


@dataclass(frozen=True)
class _WarpCase:
    """Describe one small, explicitly-toleranced resident Warp fixture."""

    name: str
    masses: np.ndarray
    concentration: np.ndarray
    gas: np.ndarray
    partitioning: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    vapor_pressure: np.ndarray
    protected_mass_mask: np.ndarray | None = None
    protected_gas_mask: np.ndarray | None = None
    time_step: float = 0.1
    latent_heat: bool = False
    mass_rtol: float = 1.0e-8
    mass_atol: float = 1.0e-30
    gas_rtol: float = 1.0e-8
    gas_atol: float = 1.0e-30
    energy_rtol: float = 1.0e-8
    energy_atol: float = 1.0e-30
    conserved: bool = True


def _configuration(*, latent_heat: bool = False) -> CondensationExecutionConfig:
    """Create the selected isothermal execution metadata."""
    return CondensationExecutionConfig(
        CondensationConfiguration(
            CondensationExecutionMode.EQUAL_STEP,
            latent_heat,
            CondensationActivityMode.IDEAL,
            CondensationSurfaceMode.STATIC,
        )
    )


def _build_legacy_cpu_state(
    case: str, time_step: float = 0.1
) -> CPUCondensationExecutionState:
    """Build a real one-box legacy aerosol state for a named CPU case."""
    vapor_pressure = 1.0e5 if case == "evaporation" else 1.0e-10
    gas_concentration = 0.0 if case == "zero_gas" else 1.0e-6
    particle_concentration = 0.0 if case == "inactive" else 1.0e6
    gas = GasSpecies(
        name="vapor",
        molar_mass=0.018,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(vapor_pressure),
        concentration=gas_concentration,
    )
    inert = GasSpecies(
        name="inert",
        molar_mass=0.029,
        vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
        partitioning=False,
        concentration=0.0,
    )
    atmosphere = (
        AtmosphereBuilder()
        .set_temperature(298.15, "K")
        .set_pressure(101325.0, "Pa")
        .set_more_partitioning_species(gas)
        .set_more_gas_only_species(inert)
        .build()
    )
    particles = (
        ResolvedParticleMassRepresentationBuilder()
        .set_distribution_strategy(ParticleResolvedSpeciatedMass())
        .set_activity_strategy(ActivityIdealMass())
        .set_surface_strategy(SurfaceStrategyVolume())
        .set_mass(np.array([[1.0e-18]], dtype=np.float64), "kg")
        .set_density(np.array([1000.0]), "kg/m^3")
        .set_charge(0)
        .set_volume(1.0, "m^3")
        .build()
    )
    particles.concentration = np.array(
        [particle_concentration], dtype=np.float64
    )
    aerosol = Aerosol(atmosphere=atmosphere, particles=particles)
    runnable = MassCondensation(
        CondensationIsothermal(
            molar_mass=np.array([0.018]),
            diffusion_coefficient=2.0e-5,
            accommodation_coefficient=1.0,
            update_gases=True,
        )
    )
    if case == "skip_partitioning":
        runnable.condensation_strategy.skip_partitioning_indices = cast(
            Any, np.array([0])
        )
    return CPUCondensationExecutionState(
        CPUCondensationState(_configuration(), aerosol), time_step, 4, runnable
    )


def _cases() -> tuple[_WarpCase, ...]:
    """Return the bounded one-/two-box direct-Warp case matrix."""
    base_mass = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    base_number = np.array([[1.0e6, 2.0e6]], dtype=np.float64)
    common: dict[str, Any] = dict(
        masses=base_mass,
        concentration=base_number,
        partitioning=np.array([[True]]),
        temperature=np.array([298.15]),
        pressure=np.array([101325.0]),
    )
    return (
        _WarpCase(
            "uptake",
            gas=np.array([[1.0e-6]]),
            vapor_pressure=np.zeros((1, 1)),
            mass_rtol=1.0e-8,
            mass_atol=1.0e-30,
            gas_rtol=1.0e-8,
            gas_atol=1.0e-30,
            **common,
        ),
        _WarpCase(
            "evaporation",
            gas=np.array([[1.0e-12]]),
            vapor_pressure=np.full((1, 1), 1.0),
            mass_rtol=1.0e-8,
            mass_atol=1.0e-30,
            gas_rtol=1.0e-8,
            gas_atol=1.0e-30,
            **common,
        ),
        _WarpCase(
            "mixed_transfer",
            masses=np.array([[[1.0e-24], [1.0e-15]]], dtype=np.float64),
            concentration=np.array([[1.0e6, 1.0e12]], dtype=np.float64),
            gas=np.zeros((1, 1)),
            partitioning=np.array([[True]]),
            temperature=np.array([298.15]),
            pressure=np.array([101325.0]),
            vapor_pressure=np.full((1, 1), 1.0e-2),
            mass_rtol=1.0e-8,
            mass_atol=1.0e-30,
            gas_rtol=1.0e-8,
            gas_atol=1.0e-30,
        ),
        _WarpCase(
            "disabled",
            gas=np.array([[1.0e-6]]),
            vapor_pressure=np.zeros((1, 1)),
            partitioning=np.array([[False]]),
            masses=base_mass,
            concentration=base_number,
            temperature=np.array([298.15]),
            pressure=np.array([101325.0]),
            mass_rtol=1.0e-8,
            mass_atol=1.0e-30,
            gas_rtol=1.0e-8,
            gas_atol=1.0e-30,
            protected_mass_mask=np.ones((1, 2, 1), dtype=bool),
            protected_gas_mask=np.ones((1, 1), dtype=bool),
            conserved=False,
        ),
        _WarpCase(
            "zero_gas",
            gas=np.zeros((1, 1)),
            vapor_pressure=np.zeros((1, 1)),
            mass_rtol=1.0e-8,
            mass_atol=1.0e-30,
            gas_rtol=1.0e-8,
            gas_atol=1.0e-30,
            protected_gas_mask=np.ones((1, 1), dtype=bool),
            **common,
        ),
        _WarpCase(
            "inactive",
            gas=np.array([[1.0e-6]]),
            vapor_pressure=np.zeros((1, 1)),
            masses=base_mass,
            concentration=np.array([[1.0e6, 0.0]]),
            partitioning=np.array([[True]]),
            temperature=np.array([298.15]),
            pressure=np.array([101325.0]),
            mass_rtol=1.0e-8,
            mass_atol=1.0e-30,
            gas_rtol=1.0e-8,
            gas_atol=1.0e-30,
            protected_mass_mask=np.array([[[False], [True]]], dtype=bool),
        ),
        _WarpCase(
            "two_box",
            masses=np.repeat(base_mass, 2, axis=0),
            concentration=np.repeat(base_number, 2, axis=0),
            gas=np.array([[1.0e-6], [2.0e-6]]),
            partitioning=np.ones((2, 1), dtype=bool),
            temperature=np.array([290.0, 310.0]),
            pressure=np.array([100000.0, 90000.0]),
            vapor_pressure=np.zeros((2, 1)),
            mass_rtol=1.0e-8,
            mass_atol=1.0e-30,
            gas_rtol=1.0e-8,
            gas_atol=1.0e-30,
        ),
        _WarpCase(
            "latent_heat",
            gas=np.array([[1.0e-6]]),
            vapor_pressure=np.zeros((1, 1)),
            latent_heat=True,
            mass_rtol=1.0e-8,
            mass_atol=1.0e-30,
            gas_rtol=1.0e-8,
            gas_atol=1.0e-30,
            energy_rtol=1.0e-8,
            energy_atol=1.0e-30,
            **common,
        ),
    )


WARP_CASES = _cases()


def _build_warp_state(
    case: _WarpCase, device: str, *, sentinel_outputs: bool = False
) -> dict[str, Any]:
    """Convert one detached float64 CPU fixture once and retain snapshots."""
    wp = pytest.importorskip("warp")
    from particula.gpu.conversion import (
        to_warp_environment_data,
        to_warp_gas_data,
        to_warp_particle_data,
    )
    from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig

    particle_data = ParticleData(
        masses=case.masses.copy(),
        concentration=case.concentration.copy(),
        charge=np.zeros_like(case.concentration),
        density=np.array([1000.0]),
        volume=np.ones(case.masses.shape[0]),
    )
    gas_data = GasData(
        name=["vapor"],
        molar_mass=np.array([0.018]),
        concentration=case.gas.copy(),
        partitioning=case.partitioning[0],
    )
    environment_data = EnvironmentData(
        temperature=case.temperature.copy(),
        pressure=case.pressure.copy(),
        saturation_ratio=np.ones_like(case.gas),
    )
    particles = to_warp_particle_data(particle_data, device=device)
    vapor_pressure = (
        np.full_like(case.vapor_pressure, 17.0)
        if sentinel_outputs
        else case.vapor_pressure
    )
    gas = to_warp_gas_data(
        gas_data, device=device, vapor_pressure=vapor_pressure
    )
    environment = to_warp_environment_data(environment_data, device=device)
    boxes, slots, species = case.masses.shape
    transfer = wp.full(
        (boxes, slots, species),
        wp.float64(7.0 if sentinel_outputs else 0.0),
        dtype=wp.float64,
        device=device,
    )
    latent_heat = (
        wp.array([2.0e5], dtype=wp.float64, device=device)
        if case.latent_heat
        else None
    )
    energy = (
        wp.full(
            (boxes, species),
            wp.float64(11.0 if sentinel_outputs else 0.0),
            dtype=wp.float64,
            device=device,
        )
        if case.latent_heat
        else None
    )
    thermal_work = wp.full(
        species,
        wp.float64(13.0 if sentinel_outputs else 0.0),
        dtype=wp.float64,
        device=device,
    )
    thermodynamics = ThermodynamicsConfig(
        modes=wp.zeros(species, dtype=wp.int32, device=device),
        parameters=wp.array(
            np.column_stack((case.vapor_pressure[0], np.zeros((species, 3)))),
            dtype=wp.float64,
            device=device,
        ),
        molar_mass_reference=wp.array([0.018], dtype=wp.float64, device=device),
    )
    state = WarpCondensationExecutionState(
        WarpCondensationState(
            _configuration(latent_heat=case.latent_heat),
            particles,
            gas,
            environment,
            thermodynamics,
            mass_transfer=transfer,
            latent_heat=latent_heat,
            energy_transfer=energy,
            thermal_work=thermal_work,
        ),
        case.time_step,
    )
    return {
        "state": state,
        "particles": particles,
        "gas": gas,
        "transfer": transfer,
        "latent_heat": latent_heat,
        "energy": energy,
        "thermal_work": thermal_work,
        "initial_mass": case.masses.copy(),
        "initial_concentration": case.concentration.copy(),
        "latent_heat_values": (
            None if not case.latent_heat else np.array([2.0e5])
        ),
        "initial_gas": case.gas.copy(),
    }


def _snapshot(
    resources: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Materialize the single post-execution host snapshot used by assertions."""
    energy = resources["energy"]
    return (
        np.array(resources["particles"].masses.numpy(), copy=True),
        np.array(resources["gas"].concentration.numpy(), copy=True),
        np.array(resources["transfer"].numpy(), copy=True),
        None if energy is None else np.array(energy.numpy(), copy=True),
    )


def _inventory(
    masses: np.ndarray, concentration: np.ndarray, gas: np.ndarray
) -> np.ndarray:
    """Return per-box, per-species concentration-weighted inventory."""
    return np.sum(masses * concentration[..., None], axis=1) + gas


def _case_message(
    case: _WarpCase, device: str, field: str, box: int, species: int
) -> str:
    """Build a failure message with case, device, field, box, and species."""
    return (
        f"case={case.name}, device={device}, field={field}, "
        f"box={box}, species={species}"
    )


def _assert_mass_and_gas(
    case: _WarpCase,
    device: str,
    masses: np.ndarray,
    gas: np.ndarray,
    expected_mass: np.ndarray,
    expected_gas: np.ndarray,
) -> None:
    """Assert case-specific particle mass and gas concentration slices."""
    for box in range(masses.shape[0]):
        for species in range(masses.shape[2]):
            npt.assert_allclose(
                masses[box, :, species],
                expected_mass[box, :, species],
                rtol=case.mass_rtol,
                atol=case.mass_atol,
                err_msg=_case_message(
                    case, device, "particle mass", box, species
                ),
            )
            npt.assert_allclose(
                gas[box, species],
                expected_gas[box, species],
                rtol=case.gas_rtol,
                atol=case.gas_atol,
                err_msg=_case_message(
                    case, device, "gas concentration", box, species
                ),
            )


def _assert_protected_lanes(
    case: _WarpCase,
    device: str,
    masses: np.ndarray,
    gas: np.ndarray,
    expected_mass: np.ndarray,
    expected_gas: np.ndarray,
) -> None:
    """Assert any declared protected particle or gas lanes remain stable."""
    if case.protected_mass_mask is not None:
        for box, slot, species in np.argwhere(case.protected_mass_mask):
            npt.assert_array_equal(
                masses[box, slot, species],
                expected_mass[box, slot, species],
                err_msg=_case_message(
                    case,
                    device,
                    "protected particle mass",
                    int(box),
                    int(species),
                ),
            )
    if case.protected_gas_mask is not None:
        for box, species in np.argwhere(case.protected_gas_mask):
            npt.assert_array_equal(
                gas[box, species],
                expected_gas[box, species],
                err_msg=_case_message(
                    case,
                    device,
                    "protected gas",
                    int(box),
                    int(species),
                ),
            )


def _p2_oracle(case: _WarpCase) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the independent fixed-four-substep direct-Warp reference.

    This deliberately evaluates the scalar physical-rate equations locally,
    rather than invoking an adapter, kernel, or GPU test helper.  The fixture
    is intentionally one-species, so ideal activity is one and the local
    calculation stays small while retaining P2 inventory limiting and coupling.
    """
    masses = case.masses.copy()
    gas = case.gas.copy()
    concentration = case.concentration
    for _ in range(4):
        proposal = np.zeros_like(masses)
        for box in range(masses.shape[0]):
            temperature = float(case.temperature[box])
            pressure = float(case.pressure[box])
            viscosity = get_dynamic_viscosity(
                temperature,
                reference_viscosity=constants.REF_VISCOSITY_AIR_STP,
                reference_temperature=constants.REF_TEMPERATURE_STP,
            )
            mean_free_path = get_molecule_mean_free_path(
                molar_mass=constants.MOLECULAR_WEIGHT_AIR,
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=viscosity,
            )
            for slot in range(masses.shape[1]):
                if concentration[box, slot] == 0.0:
                    continue
                mass = masses[box, slot, 0]
                if mass <= 0.0 or not case.partitioning[box, 0]:
                    continue
                volume = mass / 1000.0
                radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
                knudsen = get_knudsen_number(mean_free_path, radius)
                vapor_diffusion = 2.0e-5
                transport = get_first_order_mass_transport_k(
                    particle_radius=radius,
                    vapor_transition=get_vapor_transition_correction(
                        knudsen_number=knudsen,
                        mass_accommodation=1.0,
                    ),
                    diffusion_coefficient=vapor_diffusion,
                )
                kelvin = get_kelvin_term(
                    radius,
                    get_kelvin_radius(
                        effective_surface_tension=0.072,
                        effective_density=1000.0,
                        molar_mass=0.018,
                        temperature=temperature,
                    ),
                )
                surface_pressure = case.vapor_pressure[box, 0] * kelvin
                pressure_delta = (
                    get_partial_pressure(
                        concentration=gas[box, 0],
                        molar_mass=0.018,
                        temperature=temperature,
                    )
                    - surface_pressure
                )
                rate_arguments = dict(
                    pressure_delta=pressure_delta,
                    first_order_mass_transport=transport,
                    temperature=temperature,
                    molar_mass=0.018,
                )
                if case.latent_heat:
                    rate = get_mass_transfer_rate_latent_heat(
                        **rate_arguments,
                        latent_heat=2.0e5,
                        thermal_conductivity=get_thermal_conductivity(
                            temperature
                        ),
                        vapor_pressure_surface=surface_pressure,
                        diffusion_coefficient=vapor_diffusion,
                    )
                else:
                    rate = get_mass_transfer_rate(**rate_arguments)
                proposal[box, slot, 0] = rate * case.time_step / 4.0
        proposal = np.maximum(proposal, -masses)
        evaporation = np.minimum(proposal, 0.0)
        gas_after_evaporation = gas - np.sum(
            evaporation * concentration[..., None], axis=1
        )
        demand = np.sum(
            np.maximum(proposal, 0.0) * concentration[..., None], axis=1
        )
        scale = np.minimum(
            1.0,
            np.divide(
                gas_after_evaporation,
                demand,
                out=np.ones_like(gas),
                where=demand > 0.0,
            ),
        )
        proposal = np.where(
            proposal > 0.0, proposal * scale[:, None, :], proposal
        )
        masses += proposal
        gas -= np.sum(proposal * concentration[..., None], axis=1)
        np.maximum(gas, 0.0, out=gas)
    return masses, gas


def _assert_inventory(
    resources: dict[str, Any], masses: np.ndarray, gas: np.ndarray
) -> None:
    """Assert the P2 particle-plus-gas inventory is separately conserved."""
    concentration = resources["initial_concentration"]
    initial = _inventory(
        resources["initial_mass"], concentration, resources["initial_gas"]
    )
    final = _inventory(masses, concentration, gas)
    npt.assert_allclose(final, initial, rtol=1.0e-12, atol=1.0e-30)


def _assert_finalized_transfer_matches_physical_deltas(
    resources: dict[str, Any],
    masses: np.ndarray,
    gas: np.ndarray,
    transfer: np.ndarray,
    rtol: float,
    atol: float,
) -> None:
    """Assert finalized transfer equals independent particle and gas deltas."""
    concentration = np.asarray(
        resources["initial_concentration"], dtype=np.longdouble
    )
    particle_delta = np.sum(
        (
            np.asarray(masses, dtype=np.longdouble)
            - np.asarray(resources["initial_mass"], dtype=np.longdouble)
        )
        * concentration[..., None],
        axis=1,
    )
    gas_delta = np.asarray(gas, dtype=np.longdouble) - np.asarray(
        resources["initial_gas"], dtype=np.longdouble
    )
    finalized_transfer = np.sum(
        np.asarray(transfer, dtype=np.longdouble) * concentration[..., None],
        axis=1,
    )
    inventory_scale = np.sum(
        np.abs(
            np.asarray(resources["initial_mass"], dtype=np.longdouble)
            * concentration[..., None]
        ),
        axis=1,
    )
    rounding_atol = max(
        atol,
        float(np.max(inventory_scale) * np.finfo(np.float64).eps * 4.0),
    )
    npt.assert_allclose(
        finalized_transfer,
        particle_delta,
        rtol=rtol,
        atol=rounding_atol,
    )
    npt.assert_allclose(
        finalized_transfer,
        -gas_delta,
        rtol=rtol,
        atol=rounding_atol,
    )


@pytest.mark.parametrize(
    "case",
    ("uptake", "evaporation", "zero_gas", "inactive", "skip_partitioning"),
)
def test_cpu_adapter_executes_native_cases_and_conserves_inventory(
    case: str,
) -> None:
    """Native CPU adapter preserves identity and concentration-weighted mass."""
    # Keep evaporation within the legacy model's nonzero-radius regime.
    state = _build_legacy_cpu_state(
        case,
        time_step=1.0e-12 if case == "evaporation" else 0.1,
    )
    aerosol = state.state.aerosol
    initial_mass = aerosol.particles.get_species_mass(clone=True)
    concentration = aerosol.particles.get_concentration(clone=True)
    initial_gas = np.array(
        aerosol.atmosphere.partitioning_species.get_concentration(), copy=True
    )
    warning_context = (
        pytest.warns(RuntimeWarning, match="All radius values are zero")
        if case == "inactive"
        else nullcontext()
    )
    with warning_context:
        result = CPUCondensationExecutionAdapter().execute(state)
    final_mass = aerosol.particles.get_species_mass(clone=True)
    final_gas = aerosol.atmosphere.partitioning_species.get_concentration()
    assert isinstance(result, ExecutionResult)
    assert result.backend_result is not None
    assert result.state is state and result.backend_result.value is aerosol
    assert np.all(np.isfinite(final_mass)) and np.all(final_mass >= 0.0)
    npt.assert_allclose(
        np.sum(final_mass * concentration[:, None], axis=0) + final_gas,
        np.sum(initial_mass * concentration[:, None], axis=0) + initial_gas,
        rtol=1.0e-12,
        atol=1.0e-30,
    )
    if case == "zero_gas":
        # A gas-free legacy fixture may evaporate, but cannot create uptake.
        assert np.all(final_mass <= initial_mass)
        assert np.all(final_gas >= initial_gas)
    if case == "inactive":
        # Legacy mass transfer can clear this unweighted slot; it must not
        # change its zero number concentration or partitioning gas inventory.
        npt.assert_array_equal(
            aerosol.particles.get_concentration(clone=True),
            concentration,
        )
        npt.assert_array_equal(final_gas, initial_gas)
    if case == "skip_partitioning":
        npt.assert_array_equal(final_mass, initial_mass)
        npt.assert_array_equal(final_gas, initial_gas)
    if case == "uptake":
        assert np.all(final_mass > initial_mass)
        assert np.all(final_gas < initial_gas)
    if case == "evaporation":
        assert np.all(final_mass < initial_mass)
        assert np.all(final_gas > initial_gas)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize("case", WARP_CASES, ids=lambda case: case.name)
def test_warp_adapter_matches_independent_p2_oracle_and_conserves_inventory(
    case: _WarpCase,
) -> None:
    """Resident Warp result retains P2 accounting and protected-lane contract."""
    resources = _build_warp_state(case, "cpu")
    wp = pytest.importorskip("warp")
    expected_mass, expected_gas = _p2_oracle(case)
    result = WarpCondensationExecutionAdapter().execute(resources["state"])
    wp.synchronize()
    masses, gas, transfer, _ = _snapshot(resources)
    assert result.state is resources["state"]
    assert result.backend_result is not None
    backend_value = cast(tuple[Any, ...], result.backend_result.value)
    assert backend_value[0] is resources["particles"]
    _assert_mass_and_gas(case, "cpu", masses, gas, expected_mass, expected_gas)
    _assert_protected_lanes(
        case, "cpu", masses, gas, expected_mass, expected_gas
    )
    _assert_finalized_transfer_matches_physical_deltas(
        resources, masses, gas, transfer, case.mass_rtol, case.mass_atol
    )
    if case.conserved:
        _assert_inventory(resources, masses, gas)
    if case.name == "disabled":
        npt.assert_array_equal(masses, resources["initial_mass"])
        npt.assert_array_equal(gas, resources["initial_gas"])
    if case.name == "inactive":
        npt.assert_array_equal(masses[:, 1], resources["initial_mass"][:, 1])
    if case.name == "zero_gas":
        assert np.all(gas >= 0.0)
    if case.name == "mixed_transfer":
        assert masses[0, 0, 0] < resources["initial_mass"][0, 0, 0]
        assert masses[0, 1, 0] > resources["initial_mass"][0, 1, 0]
    assert np.all(masses >= 0.0)


@pytest.mark.warp
def test_warp_adapter_zero_time_is_an_exact_noop() -> None:
    """Zero time retains the resident primary state and native return value."""
    case = next(case for case in WARP_CASES if case.name == "uptake")
    resources = _build_warp_state(case, "cpu", sentinel_outputs=True)
    state = WarpCondensationExecutionState(resources["state"].state, 0.0)
    before = {
        "masses": resources["particles"].masses.numpy().copy(),
        "concentration": resources["particles"].concentration.numpy().copy(),
        "gas": resources["gas"].concentration.numpy().copy(),
        "transfer": resources["transfer"].numpy().copy(),
    }
    result = WarpCondensationExecutionAdapter().execute(state)
    pytest.importorskip("warp").synchronize()
    assert result.state is state
    assert result.backend_result is not None
    backend_value = cast(tuple[Any, ...], result.backend_result.value)
    assert backend_value[0] is resources["particles"]
    assert backend_value[1] is resources["transfer"]
    npt.assert_array_equal(
        resources["particles"].masses.numpy(), before["masses"]
    )
    npt.assert_array_equal(
        resources["particles"].concentration.numpy(), before["concentration"]
    )
    npt.assert_array_equal(
        resources["gas"].concentration.numpy(), before["gas"]
    )
    npt.assert_array_equal(resources["transfer"].numpy(), 0.0)
    npt.assert_array_equal(
        resources["gas"].vapor_pressure.numpy(), case.vapor_pressure
    )


@pytest.mark.warp
def test_warp_adapter_latent_heat_sidecars_have_identity_and_energy_accounting() -> (
    None
):
    """Resident latent sidecars retain identity and record finalized energy."""
    case = next(case for case in WARP_CASES if case.name == "latent_heat")
    resources = _build_warp_state(case, "cpu")
    result = WarpCondensationExecutionAdapter().execute(resources["state"])
    pytest.importorskip("warp").synchronize()
    masses, gas, transfer, energy = _snapshot(resources)
    assert result.backend_result is not None
    backend_value = cast(tuple[Any, ...], result.backend_result.value)
    assert backend_value[1] is resources["transfer"]
    assert resources["energy"] is resources["state"].state.energy_transfer
    assert energy is not None
    _assert_finalized_transfer_matches_physical_deltas(
        resources, masses, gas, transfer, case.mass_rtol, case.mass_atol
    )
    latent_heat_values = resources["latent_heat_values"]
    assert latent_heat_values is not None
    npt.assert_allclose(
        energy,
        np.sum(transfer, axis=1) * latent_heat_values[None, :],
        rtol=case.energy_rtol,
        atol=case.energy_atol,
    )


@pytest.mark.warp
def test_warp_adapter_real_execution_keeps_resources_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter execution neither restores resources nor synchronizes internally."""
    import particula.gpu.conversion as conversion

    resources = _build_warp_state(WARP_CASES[0], "cpu")
    wp = pytest.importorskip("warp")

    def fail(operation: str) -> Any:
        """Create an operation-specific resident-boundary failure."""

        def _fail(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise AssertionError(f"adapter initiated {operation}")

        return _fail

    with monkeypatch.context() as scoped:
        for name in (
            "to_warp_particle_data",
            "to_warp_gas_data",
            "to_warp_environment_data",
            "from_warp_particle_data",
            "from_warp_gas_data",
            "from_warp_environment_data",
        ):
            scoped.setattr(conversion, name, fail(name))
        scoped.setattr(MassCondensation, "execute", fail("CPU fallback"))
        scoped.setattr(wp, "copy", fail("copy"))
        scoped.setattr(wp, "synchronize", fail("synchronization"))
        result = WarpCondensationExecutionAdapter().execute(resources["state"])
    wp.synchronize()
    assert result.backend_result is not None
    backend_value = cast(tuple[Any, ...], result.backend_result.value)
    assert backend_value[0] is resources["particles"]
    assert backend_value[1] is resources["transfer"]


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "case",
    tuple(
        case
        for case in WARP_CASES
        if case.name in {"uptake", "two_box", "latent_heat"}
    ),
    ids=lambda case: case.name,
)
def test_warp_adapter_cuda_smoke_preserves_p2_inventory(
    case: _WarpCase,
) -> None:
    """Optional CUDA rows use the same bounded resident execution contract."""
    from particula.gpu.tests.cuda_availability import (
        CUDA_SKIP_REASON,
        cuda_available,
    )

    wp = pytest.importorskip("warp")
    if not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)
    resources = _build_warp_state(case, "cuda")
    expected_mass, expected_gas = _p2_oracle(case)
    WarpCondensationExecutionAdapter().execute(resources["state"])
    wp.synchronize()
    masses, gas, transfer, _ = _snapshot(resources)
    _assert_mass_and_gas(case, "cuda", masses, gas, expected_mass, expected_gas)
    _assert_protected_lanes(
        case, "cuda", masses, gas, expected_mass, expected_gas
    )
    _assert_inventory(resources, masses, gas)
    _assert_finalized_transfer_matches_physical_deltas(
        resources, masses, gas, transfer, case.mass_rtol, case.mass_atol
    )
