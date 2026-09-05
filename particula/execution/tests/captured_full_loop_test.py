"""P1-P3 full-loop evidence boundaries for resident graph-capture loops.

This test-only module supplies the P1 NumPy oracle and P2 uncaptured Warp-CPU
READY-path parity, conservation, forbidden-work, and no-work evidence. P3 adds
optional native-CUDA capture/replay evidence, which cleanly skips per candidate
before capture when qualification is unavailable and is never emulated on CPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import (
    Backend,
    CondensationActivityMode,
    CondensationConfiguration,
    CondensationExecutionMode,
    CondensationSurfaceMode,
    Device,
)
from particula.execution.adapters.coagulation import (
    BrownianCoagulationConfig,
    ResidentBrownianCoagulationExecutionState,
    WarpBrownianCoagulationExecutionState,
    WarpBrownianCoagulationState,
)
from particula.execution.adapters.condensation import (
    CondensationExecutionConfig,
    WarpCondensationExecutionState,
    WarpCondensationState,
)
from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationResourceShape,
    CommunicationShapeKind,
    CommunicationTransportMode,
    PrescribedVolumeUpdate,
)
from particula.execution.diagnostics import (
    ResidentDiagnosticOperation,
    ResidentDiagnosticRegistration,
    ResidentDiagnosticsPlan,
)
from particula.execution.gpu_resources import (
    CaptureResourceRequirements,
    GPUResourceRegistry,
    PreparedResourceViews,
    ResourceInventoryCapacities,
)
from particula.execution.gpu_session import (
    ResidentStepGuard,
    setup_resident_session,
)
from particula.execution.graph_capture import (
    GraphCaptureAvailability,
    GraphCaptureCapability,
    GraphCaptureLifecycleState,
    GraphCaptureNativeCallables,
    ResidentGraphCaptureBinding,
    _attach_resident_graph_capture_binding,
    capture_prepared_resident_graph,
    close_resident_graph_capture,
    create_graph_capture_lifecycle,
    create_resident_graph_capture_signature,
    qualify_prepared_resident_graph_capture,
    replay_captured_resident_graph,
)
from particula.execution.process_adapters import (
    ResidentDilutionRequest,
    ResidentNucleationRequest,
    ResidentWallLossRequest,
)
from particula.execution.resident_communication import (
    ResidentCommunicationRequest,
)
from particula.execution.resident_scheduler import (
    ResidentSimulationRequest,
    enqueue_prepared_resident_simulation,
    prepare_resident_simulation,
)
from particula.execution.state_updates import (
    ResidentEnvironmentUpdateRequest,
    ResidentGasUpdateRequest,
)
from particula.execution.tests.full_loop_test import _build_loop_fixture
from particula.execution.tests.multi_box_loop_test import (
    _binding,
    _resident_graph,
    _scheduler_request,
)
from particula.gas import EnvironmentData, GasData
from particula.gpu.kernels.nucleation import (
    NucleationConfig,
    NucleationExhaustionControls,
)
from particula.gpu.kernels.thermodynamics import ThermodynamicsConfig
from particula.gpu.tests.cuda_availability import (
    CUDA_SKIP_REASON,
    cuda_available,
)
from particula.particles import ParticleData
from particula.util.constants import GAS_CONSTANT

PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-30
_ORACLE_GAS_CONSTANT = 8.31446261815324


@dataclass(frozen=True)
class _CapturedLoopScenario:
    """Retain immutable inputs for the CPU-only captured-loop oracle."""

    logical_box_ids: np.ndarray
    particle_masses: np.ndarray
    particle_concentration: np.ndarray
    particle_charge: np.ndarray
    gas_concentration: np.ndarray
    volume: np.ndarray
    final_volume: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    molar_mass: np.ndarray
    vapor_pressure: np.ndarray
    dilution_coefficient: np.ndarray
    edge_sources: np.ndarray
    edge_destinations: np.ndarray
    edge_enabled: np.ndarray
    edge_rates: np.ndarray
    gas_partitioning: np.ndarray
    baseline_total_mass: np.ndarray
    source_ledger: np.ndarray
    sink_ledger: np.ndarray
    energy_ledger: np.ndarray
    time_step: float
    time_steps: int
    root_seed: int
    process_controls: tuple[bool, bool, bool, bool]


@dataclass(frozen=True)
class _CapturedLoopState:
    """Retain mutable detached state used exclusively by the NumPy oracle."""

    particle_masses: np.ndarray
    particle_concentration: np.ndarray
    gas_concentration: np.ndarray
    volume: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    vapor_pressure: np.ndarray
    saturation_ratio: np.ndarray


@dataclass(frozen=True)
class _CapturedLoopResult:
    """Retain detached primary, derived, and diagnostic oracle outputs."""

    state: _CapturedLoopState
    gas_concentration_snapshot: np.ndarray
    saturation_ratio_snapshot: np.ndarray
    total_species_mass: np.ndarray
    particle_number_concentration: np.ndarray
    latent_heat_energy: np.ndarray
    conservation_residual: np.ndarray


def _readonly_array(values: object, dtype: np.dtype | type) -> np.ndarray:
    """Copy an array into C-contiguous immutable storage of the required type."""
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _validate_captured_loop_scenario(  # noqa: C901
    scenario: _CapturedLoopScenario,
) -> None:
    """Validate immutable CPU-oracle fixture fields before state allocation."""
    arrays = {
        "logical_box_ids": (scenario.logical_box_ids, (2,), np.int32),
        "particle_masses": (scenario.particle_masses, (2, 3, 2), np.float64),
        "particle_concentration": (
            scenario.particle_concentration,
            (2, 3),
            np.float64,
        ),
        "particle_charge": (scenario.particle_charge, (2, 3), np.float64),
        "gas_concentration": (scenario.gas_concentration, (2, 2), np.float64),
        "volume": (scenario.volume, (2,), np.float64),
        "final_volume": (scenario.final_volume, (2,), np.float64),
        "temperature": (scenario.temperature, (2,), np.float64),
        "pressure": (scenario.pressure, (2,), np.float64),
        "molar_mass": (scenario.molar_mass, (2,), np.float64),
        "vapor_pressure": (scenario.vapor_pressure, (2,), np.float64),
        "dilution_coefficient": (
            scenario.dilution_coefficient,
            (2,),
            np.float64,
        ),
        "edge_sources": (scenario.edge_sources, (3,), np.int32),
        "edge_destinations": (scenario.edge_destinations, (3,), np.int32),
        "edge_enabled": (scenario.edge_enabled, (3,), np.bool_),
        "edge_rates": (scenario.edge_rates, (3,), np.float64),
        "gas_partitioning": (scenario.gas_partitioning, (2, 2), np.bool_),
        "baseline_total_mass": (
            scenario.baseline_total_mass,
            (2, 2),
            np.float64,
        ),
        "source_ledger": (scenario.source_ledger, (2, 2), np.float64),
        "sink_ledger": (scenario.sink_ledger, (2, 2), np.float64),
        "energy_ledger": (scenario.energy_ledger, (2, 2), np.float64),
    }
    for name, (values, shape, dtype) in arrays.items():
        if values.flags.writeable:
            raise ValueError(f"{name} must be immutable")
        if values.shape != shape:
            raise ValueError(f"{name} has invalid shape")
        if values.dtype != dtype:
            raise ValueError(f"{name} has invalid dtype")
        if not values.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
    if len(np.unique(scenario.logical_box_ids)) != 2:
        raise ValueError("logical_box_ids must be unique")
    for name in (
        "volume",
        "final_volume",
        "temperature",
        "pressure",
        "molar_mass",
        "vapor_pressure",
    ):
        if np.any(getattr(scenario, name) <= 0.0):
            raise ValueError(f"{name} must be positive")
    for name in (
        "dilution_coefficient",
        "edge_rates",
        "baseline_total_mass",
        "source_ledger",
        "sink_ledger",
        "energy_ledger",
    ):
        if np.any(getattr(scenario, name) < 0.0):
            raise ValueError(f"{name} must be nonnegative")
    for name in (
        "particle_masses",
        "particle_concentration",
        "gas_concentration",
    ):
        if np.any(getattr(scenario, name) < 0.0):
            raise ValueError(f"{name} must be nonnegative")
    enabled = scenario.edge_enabled
    if np.any(scenario.edge_sources[enabled] < 0) or np.any(
        scenario.edge_sources[enabled] >= 2
    ):
        raise ValueError("edge_sources has invalid enabled edge")
    if np.any(scenario.edge_destinations[enabled] < 0) or np.any(
        scenario.edge_destinations[enabled] >= 2
    ):
        raise ValueError("edge_destinations has invalid enabled edge")
    if isinstance(scenario.time_step, bool) or not isinstance(
        scenario.time_step, (int, float, np.integer, np.floating)
    ):
        raise ValueError("time_step must be a finite positive number")
    if not np.isfinite(scenario.time_step) or scenario.time_step <= 0.0:
        raise ValueError("time_step must be positive")
    if isinstance(scenario.time_steps, bool) or not isinstance(
        scenario.time_steps, (int, np.integer)
    ):
        raise ValueError("time_steps must be an integer")
    if scenario.time_steps < 0:
        raise ValueError("time_steps must be nonnegative")
    if isinstance(scenario.root_seed, bool) or not isinstance(
        scenario.root_seed, (int, np.integer)
    ):
        raise ValueError("root_seed must be an integer")
    if scenario.root_seed < 0:
        raise ValueError("root_seed must be nonnegative")
    if len(scenario.process_controls) != 4 or not all(
        isinstance(control, bool) for control in scenario.process_controls
    ):
        raise ValueError("process_controls must contain four booleans")


def _captured_full_loop_scenario() -> _CapturedLoopScenario:
    """Create a fresh immutable deterministic two-box full-loop fixture."""
    scenario = _CapturedLoopScenario(
        logical_box_ids=_readonly_array([101, 202], np.int32),
        particle_masses=_readonly_array(
            [
                [[1.0e-18, 2.0e-18], [0.0, 0.0], [3.0e-18, 1.0e-18]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            np.float64,
        ),
        particle_concentration=_readonly_array(
            [[2.0e6, 0.0, 5.0e5], [0.0, 0.0, 0.0]], np.float64
        ),
        particle_charge=_readonly_array(
            [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]], np.float64
        ),
        gas_concentration=_readonly_array(
            [[4.0e-9, 2.0e-9], [1.0e-9, 3.0e-9]], np.float64
        ),
        volume=_readonly_array([1.0, 2.0], np.float64),
        final_volume=_readonly_array([2.0, 1.0], np.float64),
        temperature=_readonly_array([298.15, 310.15], np.float64),
        pressure=_readonly_array([101325.0, 100000.0], np.float64),
        molar_mass=_readonly_array([0.018, 0.098], np.float64),
        vapor_pressure=_readonly_array([1000.0, 500.0], np.float64),
        dilution_coefficient=_readonly_array([0.1, 0.2], np.float64),
        edge_sources=_readonly_array([0, 1, 1], np.int32),
        edge_destinations=_readonly_array([1, 0, 0], np.int32),
        edge_enabled=_readonly_array([True, True, False], np.bool_),
        edge_rates=_readonly_array([0.1, 0.0, 0.25], np.float64),
        gas_partitioning=_readonly_array(
            [[True, True], [True, False]], np.bool_
        ),
        baseline_total_mass=_readonly_array(
            [[2.0e-9, 3.0e-9], [4.0e-9, 5.0e-9]], np.float64
        ),
        source_ledger=_readonly_array(
            [[1.0e-10, 2.0e-10], [3.0e-10, 4.0e-10]], np.float64
        ),
        sink_ledger=_readonly_array(
            [[5.0e-11, 6.0e-11], [7.0e-11, 8.0e-11]], np.float64
        ),
        energy_ledger=_readonly_array([[1.0, 2.0], [3.0, 4.0]], np.float64),
        time_step=0.5,
        time_steps=3,
        root_seed=1575,
        process_controls=(False, False, False, False),
    )
    _validate_captured_loop_scenario(scenario)
    return scenario


def _species_inventory(state: _CapturedLoopState) -> np.ndarray:
    """Calculate per-box species inventory without production helpers."""
    return state.volume[:, None] * (
        np.sum(
            state.particle_masses * state.particle_concentration[:, :, None],
            axis=1,
        )
        + state.gas_concentration
    )


def _detached_oracle_state(
    scenario: _CapturedLoopScenario,
) -> _CapturedLoopState:
    """Create writable, nonaliasing state from validated immutable input."""
    vapor_pressure = np.broadcast_to(
        scenario.vapor_pressure, scenario.gas_concentration.shape
    ).copy()
    saturation_ratio = np.zeros_like(scenario.gas_concentration)
    return _CapturedLoopState(
        particle_masses=scenario.particle_masses.copy(),
        particle_concentration=scenario.particle_concentration.copy(),
        gas_concentration=scenario.gas_concentration.copy(),
        volume=scenario.volume.copy(),
        temperature=scenario.temperature.copy(),
        pressure=scenario.pressure.copy(),
        vapor_pressure=vapor_pressure,
        saturation_ratio=saturation_ratio,
    )


def _run_captured_full_loop_oracle(
    scenario: _CapturedLoopScenario,
    steps: int,
) -> _CapturedLoopResult:
    """Advance the deterministic communication, volume, and dilution oracle."""
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    _validate_captured_loop_scenario(scenario)
    state = _detached_oracle_state(scenario)
    for _ in range(steps):
        pre_step_volume = state.volume.copy()
        amounts = state.gas_concentration * pre_step_volume[:, None]
        outbound_fractions = np.zeros_like(pre_step_volume)
        enabled = scenario.edge_enabled
        np.add.at(
            outbound_fractions,
            scenario.edge_sources[enabled],
            scenario.edge_rates[enabled] * scenario.time_step,
        )
        if not np.all(np.isfinite(outbound_fractions)):
            raise ValueError("aggregate outbound fractions must be finite")
        if np.any(outbound_fractions > 1.0):
            raise ValueError("aggregate outbound fractions must not exceed 1")
        debits = np.zeros_like(amounts)
        credits = np.zeros_like(amounts)
        for source, destination, enabled, rate in zip(
            scenario.edge_sources,
            scenario.edge_destinations,
            scenario.edge_enabled,
            scenario.edge_rates,
            strict=True,
        ):
            if enabled and rate != 0.0:
                transferred = amounts[source] * rate * scenario.time_step
                debits[source] += transferred
                credits[destination] += transferred
        state.gas_concentration[:] = (
            amounts - debits + credits
        ) / pre_step_volume[:, None]
        changed_volume = state.volume != scenario.final_volume
        if np.any(changed_volume):
            scale = (
                state.volume[changed_volume]
                / scenario.final_volume[changed_volume]
            )
            state.particle_concentration[changed_volume] *= scale[:, None]
            state.gas_concentration[changed_volume] *= scale[:, None]
            state.volume[changed_volume] = scenario.final_volume[changed_volume]
        dilution = np.exp(-scenario.dilution_coefficient * scenario.time_step)
        state.particle_concentration[:] = (
            state.particle_concentration * dilution[:, None]
        )
        state.gas_concentration[:] = state.gas_concentration * dilution[:, None]
        state.vapor_pressure[:] = scenario.vapor_pressure[None, :]
        state.saturation_ratio[:] = (
            state.gas_concentration
            * _ORACLE_GAS_CONSTANT
            * state.temperature[:, None]
            / (scenario.molar_mass[None, :] * state.vapor_pressure)
        )
    total_species_mass = _species_inventory(state)
    return _CapturedLoopResult(
        state=state,
        gas_concentration_snapshot=state.gas_concentration.copy(),
        saturation_ratio_snapshot=state.saturation_ratio.copy(),
        total_species_mass=total_species_mass.copy(),
        particle_number_concentration=np.sum(
            state.particle_concentration, axis=1
        ).copy(),
        latent_heat_energy=scenario.energy_ledger.copy(),
        conservation_residual=(
            total_species_mass
            - scenario.baseline_total_mass
            - scenario.source_ledger
            + scenario.sink_ledger
        ).copy(),
    )


def _scenario_replacement(values: object, dtype: np.dtype | type) -> np.ndarray:
    """Return a C-contiguous immutable replacement suitable for ``replace``."""
    return _readonly_array(values, dtype)


def test_captured_loop_scenario_is_fresh_immutable_and_results_detach() -> None:
    """Keep deterministic CPU-oracle inputs immutable across independent calls."""
    first = _captured_full_loop_scenario()
    second = _captured_full_loop_scenario()
    assert first.logical_box_ids.tolist() == [101, 202]
    assert first.time_steps == 3
    assert first.root_seed == 1575
    assert first.process_controls == (False, False, False, False)
    for name in (
        "particle_masses",
        "particle_concentration",
        "particle_charge",
        "gas_concentration",
        "volume",
        "final_volume",
        "temperature",
        "pressure",
        "molar_mass",
        "vapor_pressure",
        "dilution_coefficient",
    ):
        first_values = getattr(first, name)
        second_values = getattr(second, name)
        npt.assert_equal(first_values, second_values)
        assert first_values.dtype == second_values.dtype
        assert not first_values.flags.writeable
        with pytest.raises(ValueError):
            first_values.flat[0] = 0.0
    result = _run_captured_full_loop_oracle(first, 1)
    assert result.state.gas_concentration.flags.writeable
    assert not np.shares_memory(
        result.state.gas_concentration, first.gas_concentration
    )
    assert not np.shares_memory(
        result.total_species_mass, first.baseline_total_mass
    )


def test_captured_loop_oracle_matches_one_and_multiple_step_literals() -> None:
    """Check primary and derived fields against explicit two-box calculations."""
    scenario = _captured_full_loop_scenario()
    one_step = _run_captured_full_loop_oracle(scenario, 1)
    first_dilution = np.exp(-0.05)
    second_dilution = np.exp(-0.1)
    npt.assert_allclose(
        one_step.state.particle_concentration,
        [[1.0e6 * first_dilution, 0.0, 2.5e5 * first_dilution], [0.0] * 3],
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )
    npt.assert_allclose(
        one_step.state.gas_concentration,
        [
            [1.9e-9 * first_dilution, 0.95e-9 * first_dilution],
            [2.2e-9 * second_dilution, 6.1e-9 * second_dilution],
        ],
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )
    npt.assert_allclose(one_step.state.volume, [2.0, 1.0])
    npt.assert_equal(one_step.state.particle_masses, scenario.particle_masses)
    npt.assert_equal(one_step.state.temperature, scenario.temperature)
    npt.assert_equal(one_step.state.pressure, scenario.pressure)
    npt.assert_allclose(one_step.state.vapor_pressure, [[1000.0, 500.0]] * 2)
    expected_saturation = (
        np.array([[1.9e-9, 0.95e-9], [2.2e-9, 6.1e-9]])
        * np.array([[first_dilution], [second_dilution]])
        * _ORACLE_GAS_CONSTANT
        * scenario.temperature[:, None]
        / (scenario.molar_mass[None, :] * scenario.vapor_pressure[None, :])
    )
    npt.assert_allclose(
        one_step.state.saturation_ratio,
        expected_saturation,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )
    multiple = _run_captured_full_loop_oracle(scenario, 3)
    npt.assert_allclose(
        multiple.state.particle_concentration[0],
        np.array([1.0e6, 0.0, 2.5e5]) * np.exp(-0.15),
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )
    npt.assert_equal(multiple.state.particle_concentration[1], 0.0)
    npt.assert_allclose(multiple.state.volume, [2.0, 1.0])


def test_captured_loop_oracle_reports_independent_diagnostics() -> None:
    """Compare every result diagnostic with an independent direct expression."""
    scenario = _captured_full_loop_scenario()
    result = _run_captured_full_loop_oracle(scenario, 2)
    state = result.state
    inventory = state.volume[:, None] * (
        np.sum(
            state.particle_masses * state.particle_concentration[:, :, None],
            axis=1,
        )
        + state.gas_concentration
    )
    npt.assert_allclose(
        result.gas_concentration_snapshot, state.gas_concentration
    )
    npt.assert_allclose(
        result.saturation_ratio_snapshot, state.saturation_ratio
    )
    npt.assert_allclose(result.total_species_mass, inventory)
    npt.assert_allclose(
        result.particle_number_concentration,
        np.sum(state.particle_concentration, axis=1),
    )
    npt.assert_equal(result.latent_heat_energy, scenario.energy_ledger)
    npt.assert_allclose(
        result.conservation_residual,
        inventory
        - scenario.baseline_total_mass
        - scenario.source_ledger
        + scenario.sink_ledger,
    )


def test_captured_loop_inventory_separates_transport_volume_and_dilution() -> (
    None
):
    """Prevent dilution loss from being misreported as communication loss."""
    scenario = _captured_full_loop_scenario()
    no_dilution = replace(
        scenario,
        dilution_coefficient=_scenario_replacement([0.0, 0.0], np.float64),
    )
    initial = _detached_oracle_state(no_dilution)
    result = _run_captured_full_loop_oracle(no_dilution, 1)
    npt.assert_allclose(
        np.sum(_species_inventory(result.state), axis=0),
        np.sum(_species_inventory(initial), axis=0),
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )
    isolated = replace(
        no_dilution,
        edge_enabled=_scenario_replacement([False, False, False], np.bool_),
    )
    isolated_result = _run_captured_full_loop_oracle(isolated, 1)
    npt.assert_allclose(
        _species_inventory(isolated_result.state),
        _species_inventory(_detached_oracle_state(isolated)),
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )
    diluted = _run_captured_full_loop_oracle(scenario, 1)
    npt.assert_allclose(
        _species_inventory(diluted.state),
        _species_inventory(result.state)
        * np.exp(-scenario.dilution_coefficient[:, None] * scenario.time_step),
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
    )


def test_captured_loop_gas_work_oracle_has_per_dispatch_ledgers() -> None:
    """Keep detached GAS work expectations separate from accumulated state."""
    scenario = _captured_full_loop_scenario()
    amounts, deltas, outbound = _independent_gas_work_oracle(
        scenario.gas_concentration,
        scenario.volume,
        scenario.edge_sources,
        scenario.edge_destinations,
        scenario.edge_enabled,
        scenario.edge_rates,
        scenario.time_step,
    )
    npt.assert_allclose(
        amounts,
        [[4.0e-9, 2.0e-9], [2.0e-9, 6.0e-9]],
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg="communication amounts",
    )
    npt.assert_allclose(
        deltas,
        [[-2.0e-10, -1.0e-10], [2.0e-10, 1.0e-10]],
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg="communication amount deltas",
    )
    npt.assert_allclose(
        outbound,
        [[2.0e-10, 1.0e-10], [0.0, 0.0]],
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg="communication outbound amounts",
    )


def test_captured_loop_no_work_rows_and_inactive_slots_are_write_free() -> None:
    """Keep disabled transport, zero rates, and empty particle rows canonical."""
    scenario = _captured_full_loop_scenario()
    no_work = replace(
        scenario,
        edge_enabled=_scenario_replacement([False, True, False], np.bool_),
        edge_rates=_scenario_replacement([0.0, 0.0, 0.25], np.float64),
        dilution_coefficient=_scenario_replacement([0.0, 0.0], np.float64),
        final_volume=_scenario_replacement([1.0, 2.0], np.float64),
    )
    result = _run_captured_full_loop_oracle(no_work, 1)
    npt.assert_equal(
        result.state.particle_concentration, scenario.particle_concentration
    )
    npt.assert_equal(result.state.gas_concentration, scenario.gas_concentration)
    npt.assert_equal(result.state.volume, scenario.volume)
    npt.assert_equal(result.state.particle_masses[0, 1], [0.0, 0.0])
    npt.assert_equal(result.state.particle_concentration[1], [0.0, 0.0, 0.0])
    assert np.all(np.isfinite(result.saturation_ratio_snapshot))


def test_captured_loop_saturation_snapshot_uses_final_gas_state() -> None:
    """Derive saturation after communication, volume, and dilution updates."""
    scenario = replace(
        _captured_full_loop_scenario(),
        edge_enabled=_scenario_replacement([False, False, False], np.bool_),
        final_volume=_scenario_replacement([1.0, 2.0], np.float64),
    )
    result = _run_captured_full_loop_oracle(scenario, 1)
    expected_saturation = (
        scenario.gas_concentration
        * np.exp(-scenario.dilution_coefficient[:, None] * scenario.time_step)
        * _ORACLE_GAS_CONSTANT
        * scenario.temperature[:, None]
        / (scenario.molar_mass[None, :] * scenario.vapor_pressure[None, :])
    )
    npt.assert_allclose(result.saturation_ratio_snapshot, expected_saturation)
    npt.assert_allclose(
        result.gas_concentration_snapshot,
        scenario.gas_concentration
        * np.exp(-scenario.dilution_coefficient[:, None] * scenario.time_step),
    )


@pytest.mark.parametrize(
    ("steps", "exception", "message"),
    [
        (None, TypeError, "integer"),
        ("1", TypeError, "integer"),
        (1.0, TypeError, "integer"),
        (True, TypeError, "integer"),
        (-1, ValueError, "nonnegative"),
    ],
)
def test_captured_loop_invalid_steps_preserve_immutable_scenario(
    steps: object,
    exception: type[Exception],
    message: str,
) -> None:
    """Reject invalid step counts before allocating or changing private state."""
    scenario = _captured_full_loop_scenario()
    before = {
        name: getattr(scenario, name).copy()
        for name in (
            "particle_masses",
            "particle_concentration",
            "gas_concentration",
            "volume",
        )
    }
    with pytest.raises(exception, match=message):
        _run_captured_full_loop_oracle(scenario, steps)  # type: ignore[arg-type]
    for name, values in before.items():
        npt.assert_equal(getattr(scenario, name), values)
        assert not getattr(scenario, name).flags.writeable


@pytest.mark.parametrize(
    ("field", "values", "dtype", "message"),
    [
        (
            "gas_concentration",
            np.zeros((1, 2)),
            np.float64,
            "gas_concentration",
        ),
        (
            "gas_concentration",
            np.zeros((2, 2)),
            np.float32,
            "gas_concentration",
        ),
        (
            "particle_concentration",
            np.zeros((3, 2)).T,
            np.float64,
            "particle_concentration",
        ),
        ("logical_box_ids", [101, 101], np.int32, "logical_box_ids"),
        ("temperature", [np.nan, 300.0], np.float64, "temperature"),
        ("temperature", [np.inf, 300.0], np.float64, "temperature"),
        ("pressure", [0.0, 100000.0], np.float64, "pressure"),
        (
            "dilution_coefficient",
            [-0.1, 0.0],
            np.float64,
            "dilution_coefficient",
        ),
        ("edge_rates", [-0.1, 0.0, 0.0], np.float64, "edge_rates"),
        ("edge_sources", [2, 1, 1], np.int32, "edge_sources"),
    ],
)
def test_captured_loop_malformed_scenarios_reject_without_source_mutation(
    field: str,
    values: object,
    dtype: np.dtype | type,
    message: str,
) -> None:
    """Name malformed fixture fields and leave the original fixture unchanged."""
    scenario = _captured_full_loop_scenario()
    original_arrays = {
        name: value.copy()
        for name, value in vars(scenario).items()
        if isinstance(value, np.ndarray)
    }
    replacement = np.array(values, dtype=dtype, order="C", copy=True)
    if field == "particle_concentration":
        replacement = replacement.T
    replacement.setflags(write=False)
    malformed = replace(scenario, **{field: replacement})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        _run_captured_full_loop_oracle(malformed, 1)
    for name, original in original_arrays.items():
        npt.assert_equal(getattr(scenario, name), original)
        assert not getattr(scenario, name).flags.writeable


@pytest.mark.parametrize(
    ("field", "index", "value", "message"),
    [
        ("particle_masses", (0, 0, 0), -1.0, "particle_masses"),
        ("particle_concentration", (0, 0), -1.0, "particle_concentration"),
        ("gas_concentration", (0, 0), -1.0, "gas_concentration"),
    ],
)
def test_captured_loop_rejects_negative_primary_state(
    field: str,
    index: tuple[int, ...],
    value: float,
    message: str,
) -> None:
    """Reject invalid primary state without mutating the source scenario."""
    scenario = _captured_full_loop_scenario()
    replacement = getattr(scenario, field).copy()
    replacement.setflags(write=True)
    replacement[index] = value
    replacement.setflags(write=False)
    malformed = replace(scenario, **{field: replacement})
    with pytest.raises(ValueError, match=message):
        _run_captured_full_loop_oracle(malformed, 1)
    assert getattr(scenario, field)[index] >= 0.0


def test_captured_loop_rejects_writable_scenario_arrays() -> None:
    """Require all scenario-owned arrays to remain immutable before execution."""
    scenario = _captured_full_loop_scenario()
    writable_gas = scenario.gas_concentration.copy()
    malformed = replace(scenario, gas_concentration=writable_gas)
    with pytest.raises(ValueError, match="gas_concentration must be immutable"):
        _run_captured_full_loop_oracle(malformed, 1)
    assert writable_gas.flags.writeable
    assert not scenario.gas_concentration.flags.writeable


@pytest.mark.parametrize(
    ("rates", "message"),
    [
        ([np.finfo(np.float64).max] * 3, "finite"),
        ([1.5, 1.5, 0.0], "must not exceed 1"),
    ],
)
def test_captured_loop_rejects_invalid_aggregate_transport_fraction(
    rates: list[float], message: str
) -> None:
    """Reject nonfinite or overdraw transport before oracle state commits."""
    scenario = _captured_full_loop_scenario()
    malformed = replace(
        scenario,
        edge_enabled=_scenario_replacement([True, True, True], np.bool_),
        edge_sources=_scenario_replacement([0, 0, 0], np.int32),
        edge_destinations=_scenario_replacement([1, 1, 0], np.int32),
        edge_rates=_scenario_replacement(rates, np.float64),
    )
    if message == "finite":
        with pytest.warns(RuntimeWarning, match="overflow encountered in add"):
            with pytest.raises(ValueError, match=message):
                _run_captured_full_loop_oracle(malformed, 1)
    else:
        with pytest.raises(ValueError, match=message):
            _run_captured_full_loop_oracle(malformed, 1)
    npt.assert_equal(malformed.gas_concentration, scenario.gas_concentration)


@dataclass
class _PreparedLoop:
    """Retain one authentic prepared resident loop and its resources."""

    wp: Any
    session: Any
    registry: Any
    guard: Any
    request: Any
    binding: ResidentGraphCaptureBinding
    prepared: Any
    coagulation_resources: Any
    wall_loss_resources: Any


class _WarpNativeCaptureAdapter:
    """Expose Warp's genuine native graph operations after the CUDA gate."""

    def __init__(self, wp: Any, native: str) -> None:
        self.wp = wp
        self.native = native

    def runtime_available(self) -> bool:
        """Return the already-established runtime qualification."""
        return True

    def device_available(self, device: Device) -> bool:
        """Require the exact pre-qualified native CUDA device."""
        return device.native == self.native

    def capture_api_available(self, device: Device) -> bool:
        """Require the exact device whose APIs were checked before setup."""
        return device.native == self.native

    def capture_callables(self, device: Device) -> GraphCaptureNativeCallables:
        """Bind Warp capture, replay, and exact-handle cleanup operations."""

        def capture_begin() -> None:
            self.wp.capture_begin(
                device=device.native,
                force_module_load=True,
            )

        def capture_release(handle: object) -> None:
            destroy = getattr(handle, "destroy", None)
            if callable(destroy):
                destroy()
            # Warp versions without a public destroy method release the native
            # graph when this exact opaque Python handle loses its final owner.

        return GraphCaptureNativeCallables(
            capture_begin,
            self.wp.capture_end,
            lambda: None,
            self.wp.capture_launch,
            capture_release,
        )


@pytest.fixture(autouse=True)
def _remove_resolver_schedule_registrations() -> Any:
    """Keep locally constructed schedules isolated between test cases."""
    import particula.execution.scheduler as scheduler_module

    schedules = scheduler_module._RESOLVER_SCHEDULES
    initial_schedules = list(schedules)
    _resident_graph.cache_clear()
    schedules.clear()
    yield
    _resident_graph.cache_clear()
    schedules[:] = initial_schedules


def _inventory_oracle(fixture: Any) -> np.ndarray:
    """Compute concentration-weighted inventory independently."""
    particles = fixture.session.particles
    gas = fixture.session.gas
    return particles.volume.numpy()[:, None] * (
        np.sum(
            particles.masses.numpy()
            * particles.concentration.numpy()[:, :, None],
            axis=1,
        )
        + gas.concentration.numpy()
    )


class _FakeNativeCapture:
    """Record native capture calls without emulating CUDA on Warp CPU."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.launches: list[object] = []
        self.handle = object()

    def callables(self) -> GraphCaptureNativeCallables:
        """Return the complete opaque native vocabulary used by P1--P3."""

        def capture_begin() -> None:
            self.calls.append("begin")

        def capture_end() -> object:
            self.calls.append("end")
            return self.handle

        def capture_instantiate() -> None:
            self.calls.append("instantiate")

        def capture_launch(handle: object) -> None:
            self.calls.append("launch")
            self.launches.append(handle)

        def capture_release(handle: object) -> None:
            self.calls.append("release")
            assert handle is self.handle

        return GraphCaptureNativeCallables(
            capture_begin,
            capture_end,
            capture_instantiate,
            capture_launch,
            capture_release,
        )

    def adapter(self) -> Any:
        """Return an adapter with deterministic ordered availability probes."""
        native = self.callables()
        owner = self

        class Adapter:
            """Expose the fake runtime and native callable vocabulary."""

            def runtime_available(self) -> bool:
                owner.calls.append("runtime")
                return True

            def device_available(self, device: Device) -> bool:
                owner.calls.append(f"device:{device.native}")
                return True

            def capture_api_available(self, device: Device) -> bool:
                owner.calls.append(f"api:{device.native}")
                return True

            def capture_callables(
                self, device: Device
            ) -> GraphCaptureNativeCallables:
                owner.calls.append(f"callables:{device.native}")
                return native

        return Adapter()


def _prepared_fake_capture(
    fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    duration: float,
) -> tuple[Any, Any, _FakeNativeCapture]:
    """Build an exact READY prepared binding and qualify it on fake CUDA."""
    import particula.execution.resident_scheduler as resident_scheduler

    request = fixture.request
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            request.session.metadata.device,
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request, request.session, request.registry, request.guard, lifecycle
    )
    _attach_resident_graph_capture_binding(request, binding)
    object.__setattr__(request.condensation, "time_step", duration)
    object.__setattr__(request.coagulation.request.state, "time_step", duration)
    object.__setattr__(request.dilution, "time_step", duration)
    object.__setattr__(request.wall_loss, "time_step", duration)
    object.__setattr__(request.nucleation, "time_step", duration)
    object.__setattr__(request.communication, "duration", duration)

    class PreparedAdapter:
        """Provide a write-free prepared operation for capture setup."""

        def prepare(self, _request: object) -> Any:
            return SimpleNamespace(execute=lambda: None)

    for name in (
        "WarpCondensationExecutionAdapter",
        "ResidentBrownianCoagulationExecutionAdapter",
        "ResidentDilutionAdapter",
        "ResidentWallLossAdapter",
        "ResidentNucleationAdapter",
    ):
        monkeypatch.setattr(resident_scheduler, name, PreparedAdapter)
    prepared = resident_scheduler.prepare_resident_simulation(request, duration)
    capture_set = fixture.registry.validate_capture_resource_set(
        request.capture_resource_requirements
    )

    cuda_device = Device(Backend.WARP, "cuda:0")
    object.__setattr__(request.session.metadata, "device", cuda_device)
    object.__setattr__(signature, "device", cuda_device)
    object.__setattr__(lifecycle.capability, "device", cuda_device)
    monkeypatch.setattr(
        request.registry, "validate_pinned_session", lambda _session: None
    )
    native = _FakeNativeCapture()
    qualification = qualify_prepared_resident_graph_capture(
        binding, prepared, capture_set, native.adapter()
    )
    return qualification, binding, native


def _snapshot(fixture: Any) -> tuple[np.ndarray, ...]:
    """Synchronize at an assertion boundary and copy meaningful state."""
    fixture.wp.synchronize()
    session = fixture.session
    return (
        session.particles.masses.numpy().copy(),
        session.particles.concentration.numpy().copy(),
        session.particles.charge.numpy().copy(),
        session.particles.density.numpy().copy(),
        session.particles.volume.numpy().copy(),
        session.gas.concentration.numpy().copy(),
        session.gas.vapor_pressure.numpy().copy(),
        session.gas.partitioning.numpy().copy(),
        session.environment.temperature.numpy().copy(),
        session.environment.pressure.numpy().copy(),
        session.environment.saturation_ratio.numpy().copy(),
        fixture.gas_snapshot.numpy().copy(),
        fixture.saturation_snapshot.numpy().copy(),
        fixture.request.coagulation.resources.rng_states.numpy().copy(),
        fixture.request.wall_loss.resources.rng_states.numpy().copy(),
    )


def _zero_nucleation_request(
    session: Any,
    registry: Any,
    resources: Any,
    duration: float,
) -> ResidentNucleationRequest:
    """Build a valid, deterministic no-admission nucleation request."""
    config = NucleationConfig(
        rate_law="activation",
        coefficient=0.0,
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1, 0),
        formation_diameter=1.0e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1.0e40,
        temperature_lower=100.0,
        temperature_upper=500.0,
    )
    return ResidentNucleationRequest(
        session,
        registry,
        resources,
        config,
        duration,
        NucleationExhaustionControls(True, False),
    )


def _capture_communication_configuration(session: Any, wp: Any) -> Any:
    """Build a closed, zero-rate one-dimensional map on the active device."""
    device = session.particles.masses.device
    edges = max(session.dimensions.n_boxes - 1, 0)
    sources = np.arange(edges, dtype=np.int32)
    destinations = sources + 1
    return CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ONE_DIMENSIONAL,
            CommunicationTransportMode.GAS,
            edges,
            wp.array(sources, dtype=wp.int32, device=device),
            wp.array(destinations, dtype=wp.int32, device=device),
            wp.ones(edges, dtype=wp.int32, device=device),
            wp.zeros(edges, dtype=wp.float64, device=device),
        ),
        PrescribedVolumeUpdate(None),
        (
            CommunicationResourceShape(
                "edge_rates",
                wp.float64,
                CommunicationShapeKind.E,
            ),
        ),
    )


def _build_prepared_loop(
    device: str,
    n_boxes: int,
    duration: float,
    root_seed: int,
    *,
    selected_wall_loss_boxes: tuple[int, ...] = (),
    single_active_particle: bool = False,
) -> _PreparedLoop:
    """Build one real all-operation resident loop on the requested device."""
    manifest = tuple((f"box-{2 * index}", index) for index in range(n_boxes))
    session, registry, guard = _binding(device, manifest, root_seed)
    wp = pytest.importorskip("warp")
    if single_active_particle:
        warp_device = session.particles.masses.device
        masses = session.particles.masses.numpy()
        concentration = session.particles.concentration.numpy()
        charge = session.particles.charge.numpy()
        masses[:, 1:, :] = 0.0
        concentration[:, 1:] = 0.0
        charge[:, 1:] = 0.0
        wp.copy(
            session.particles.masses,
            wp.array(masses, dtype=wp.float64, device=warp_device),
        )
        wp.copy(
            session.particles.concentration,
            wp.array(concentration, dtype=wp.float64, device=warp_device),
        )
        wp.copy(
            session.particles.charge,
            wp.array(charge, dtype=wp.float64, device=warp_device),
        )
    _resident_graph.cache_clear()
    request, wall_loss, coagulation = _scheduler_request(
        session,
        registry,
        guard,
        duration,
        selected_wall_loss_boxes,
        _capture_communication_configuration(session, wp),
        session.dimensions.n_particles,
    )
    request = replace(
        request,
        nucleation=_zero_nucleation_request(
            session,
            registry,
            request.nucleation.resources,
            duration,
        ),
    )
    signature = create_resident_graph_capture_signature(request)
    lifecycle = create_graph_capture_lifecycle(
        GraphCaptureCapability(
            session.metadata.device,
            GraphCaptureAvailability.AVAILABLE,
        ),
        signature,
    )
    binding = ResidentGraphCaptureBinding(
        request,
        session,
        registry,
        guard,
        lifecycle,
    )
    _attach_resident_graph_capture_binding(request, binding)
    prepared = prepare_resident_simulation(request, duration)
    return _PreparedLoop(
        wp=wp,
        session=session,
        registry=registry,
        guard=guard,
        request=request,
        binding=binding,
        prepared=prepared,
        coagulation_resources=coagulation,
        wall_loss_resources=wall_loss,
    )


@dataclass
class _ScenarioPreparedLoop:
    """Retain the P1 scenario's exact READY binding and owned test arrays."""

    loop: _PreparedLoop
    communication_resources: Any
    latent_energy: Any
    baseline_total_mass: Any
    source_ledger: Any
    sink_ledger: Any


def _no_op_prepared_state_update(*_prepared: object) -> None:
    """Keep scenario state-update sources metadata-only across loop builds."""


def _build_scenario_prepared_loop_impl(
    monkeypatch: pytest.MonkeyPatch,
    duration: float,
    cleanup_binding: list[tuple[Any, Any, Any]],
    device: Device = Device(Backend.WARP, "cpu"),
    communication_family: CommunicationTransportMode = (
        CommunicationTransportMode.GAS
    ),
    scenario: _CapturedLoopScenario | None = None,
) -> _ScenarioPreparedLoop:
    """Build the P1 two-box scenario without shared fixture update sources."""
    import particula.execution.resident_scheduler as resident_scheduler
    from particula.execution.resident_scheduler import _CANONICAL_IDS
    from particula.gpu.kernels.wall_loss import NeutralWallLossConfig

    wp = pytest.importorskip("warp")
    if scenario is None:
        scenario = _captured_full_loop_scenario()
    particles = ParticleData(
        masses=scenario.particle_masses.copy(),
        concentration=scenario.particle_concentration.copy(),
        charge=scenario.particle_charge.copy(),
        density=np.array([1000.0, 1200.0], dtype=np.float64),
        volume=scenario.volume.copy(),
    )
    gas = GasData(
        name=["species-a", "species-b"],
        molar_mass=scenario.molar_mass.copy(),
        concentration=scenario.gas_concentration.copy(),
        partitioning=scenario.gas_partitioning[0].copy(),
    )
    environment = EnvironmentData(
        temperature=scenario.temperature.copy(),
        pressure=scenario.pressure.copy(),
        saturation_ratio=np.zeros_like(scenario.gas_concentration),
    )
    session = setup_resident_session(
        particles,
        gas,
        environment,
        device,
        root_seed=scenario.root_seed,
        logical_box_ids=tuple(str(value) for value in scenario.logical_box_ids),
        lanes=(0, 1),
    )
    registry = GPUResourceRegistry(session)
    guard = ResidentStepGuard(session, registry)
    cleanup_binding.append((session, registry, guard))
    graph, schedule, by_id = _resident_graph()
    warp_device = cast(Any, session.particles).masses.device
    configuration = CommunicationConfiguration(
        CommunicationMap(
            CommunicationMapForm.ONE_DIMENSIONAL,
            communication_family,
            len(scenario.edge_sources),
            wp.array(scenario.edge_sources, dtype=wp.int32, device=warp_device),
            wp.array(
                scenario.edge_destinations, dtype=wp.int32, device=warp_device
            ),
            wp.array(scenario.edge_enabled, dtype=wp.int32, device=warp_device),
            wp.array(scenario.edge_rates, dtype=wp.float64, device=warp_device),
        ),
        PrescribedVolumeUpdate(
            wp.array(
                scenario.volume if duration == 0.0 else scenario.final_volume,
                dtype=wp.float64,
                device=warp_device,
            )
        ),
        (
            CommunicationResourceShape(
                "edge_rates", wp.float64, CommunicationShapeKind.E
            ),
        ),
    )
    communication = registry.acquire_communication(configuration)
    condensation_resources = registry.acquire_condensation()
    coagulation_resources = registry.acquire_coagulation(1)
    wall_loss_resources = registry.acquire_wall_loss()
    nucleation_resources = registry.acquire_nucleation()
    thermodynamics = ThermodynamicsConfig(
        modes=wp.zeros(2, dtype=wp.int32, device=warp_device),
        parameters=wp.array(
            np.column_stack(
                (scenario.vapor_pressure, np.zeros((2, 3), dtype=np.float64))
            ),
            dtype=wp.float64,
            device=warp_device,
        ),
        molar_mass_reference=wp.array(
            scenario.molar_mass, dtype=wp.float64, device=warp_device
        ),
    )
    condensation = WarpCondensationExecutionState(
        WarpCondensationState(
            CondensationExecutionConfig(
                CondensationConfiguration(
                    CondensationExecutionMode.EQUAL_STEP,
                    False,
                    CondensationActivityMode.IDEAL,
                    CondensationSurfaceMode.STATIC,
                )
            ),
            session.particles,
            session.gas,
            session.environment,
            thermodynamics,
            scratch_buffers=condensation_resources.scratch_buffers,
        ),
        duration,
    )
    coagulation = ResidentBrownianCoagulationExecutionState(
        WarpBrownianCoagulationExecutionState(
            WarpBrownianCoagulationState(
                BrownianCoagulationConfig(),
                session.particles,
                None,
                None,
                duration,
                collision_pairs=coagulation_resources.collision_pairs,
                n_collisions=coagulation_resources.n_collisions,
                rng_states=coagulation_resources.rng_states,
                initialize_rng=False,
                environment=session.environment,
            )
        ),
        session,
        registry,
        coagulation_resources,
    )

    def matrix() -> Any:
        """Allocate one P1 diagnostic matrix on the resident device."""
        return wp.zeros((2, 2), dtype=wp.float64, device=warp_device)

    latent_energy = matrix()
    baseline_total_mass = wp.array(
        scenario.baseline_total_mass, dtype=wp.float64, device=warp_device
    )
    source_ledger = wp.array(
        scenario.source_ledger, dtype=wp.float64, device=warp_device
    )
    sink_ledger = wp.array(
        scenario.sink_ledger, dtype=wp.float64, device=warp_device
    )
    diagnostics = ResidentDiagnosticsPlan(
        session,
        registry,
        graph,
        schedule,
        by_id["diagnostics"],
        (
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.GAS_CONCENTRATION_SNAPSHOT, matrix()
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.SATURATION_RATIO_SNAPSHOT, matrix()
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.TOTAL_SPECIES_MASS, matrix()
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION,
                wp.zeros(2, dtype=wp.float64, device=warp_device),
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.LATENT_HEAT_ENERGY,
                latent_energy,
                energy_transfer=wp.array(
                    scenario.energy_ledger, dtype=wp.float64, device=warp_device
                ),
            ),
            ResidentDiagnosticRegistration(
                ResidentDiagnosticOperation.CONSERVATION_RESIDUAL,
                matrix(),
                baseline_total_mass=baseline_total_mass,
                source_ledger=source_ledger,
                sink_ledger=sink_ledger,
            ),
        ),
    )
    inventory = registry.register_capture_resources(
        session, communication, diagnostics.registrations
    )
    requirements = CaptureResourceRequirements(
        session,
        ResourceInventoryCapacities(
            1,
            len(scenario.edge_sources)
            if communication_family is CommunicationTransportMode.GAS
            else 0,
            len(scenario.edge_sources)
            if communication_family is CommunicationTransportMode.PARTICLES
            else 0,
        ),
        inventory,
        PreparedResourceViews(
            condensation_resources,
            coagulation_resources,
            None,
            wall_loss_resources,
            nucleation_resources,
        ),
        communication,
        ("condensation", "coagulation", "wall_loss", "nucleation", "dilution"),
        condensation_resources.scratch_buffers,
        coagulation_resources,
        wall_loss_resources,
        nucleation_resources,
    )
    published = registry.prepare_capture_resources(requirements)
    assert published.dilution is not None
    wp.copy(
        published.dilution.normalized_coefficient,
        wp.array(
            scenario.dilution_coefficient, dtype=wp.float64, device=warp_device
        ),
    )
    request = ResidentSimulationRequest(
        session,
        registry,
        guard,
        graph,
        schedule,
        thermodynamics,
        condensation,
        coagulation,
        ResidentDilutionRequest(
            session,
            registry,
            published.dilution.normalized_coefficient,
            duration,
            published.dilution,
        ),
        ResidentWallLossRequest(
            session,
            registry,
            wall_loss_resources,
            NeutralWallLossConfig("spherical", 1.0, chamber_radius=1.0),
            duration,
            enabled_box_indices=(),
        ),
        ResidentNucleationRequest(
            session,
            registry,
            nucleation_resources,
            object(),
            duration,
            object(),
        ),
        diagnostics,
        ResidentEnvironmentUpdateRequest(
            session,
            registry,
            graph,
            by_id["environment_update"],
            wp.array(
                scenario.temperature, dtype=wp.float64, device=warp_device
            ),
            wp.array(scenario.pressure, dtype=wp.float64, device=warp_device),
        ),
        ResidentGasUpdateRequest(
            session,
            registry,
            graph,
            by_id["gas_update"],
            wp.array(
                scenario.gas_concentration, dtype=wp.float64, device=warp_device
            ),
        ),
        ResidentCommunicationRequest(
            session,
            registry,
            graph,
            communication,
            by_id["communication"],
            by_id["volume_evolution"],
            duration,
        ),
        requirements,
    )
    signature = create_resident_graph_capture_signature(request)
    binding = ResidentGraphCaptureBinding(
        request,
        session,
        registry,
        guard,
        create_graph_capture_lifecycle(
            GraphCaptureCapability(
                session.metadata.device, GraphCaptureAvailability.AVAILABLE
            ),
            signature,
        ),
    )
    _attach_resident_graph_capture_binding(request, binding)

    class NoOpAdapter:
        """Retain a test-only write-free process operation."""

        def prepare(self, _request: object) -> Any:
            """Return the frozen no-op prepared operation."""
            return SimpleNamespace(execute=lambda: None)

    for name in (
        "WarpCondensationExecutionAdapter",
        "ResidentBrownianCoagulationExecutionAdapter",
        "ResidentWallLossAdapter",
        "ResidentNucleationAdapter",
    ):
        monkeypatch.setattr(resident_scheduler, name, NoOpAdapter)

    monkeypatch.setattr(
        resident_scheduler,
        "setup_prepared_environment_update",
        lambda _prepared, _request: None,
    )
    monkeypatch.setattr(
        resident_scheduler,
        "setup_prepared_gas_update",
        lambda _prepared, _request: None,
    )
    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_prepared_environment_update",
        _no_op_prepared_state_update,
    )
    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_prepared_gas_update",
        _no_op_prepared_state_update,
    )
    prepared = prepare_resident_simulation(request, duration)
    assert prepared.ordered_node_ids == _CANONICAL_IDS
    loop = _PreparedLoop(
        wp,
        session,
        registry,
        guard,
        request,
        binding,
        prepared,
        coagulation_resources,
        wall_loss_resources,
    )
    result = _ScenarioPreparedLoop(
        loop,
        communication,
        latent_energy,
        baseline_total_mass,
        source_ledger,
        sink_ledger,
    )
    cleanup_binding.clear()
    return result


def _build_scenario_prepared_loop(
    monkeypatch: pytest.MonkeyPatch,
    duration: float,
    device: Device = Device(Backend.WARP, "cpu"),
    communication_family: CommunicationTransportMode = (
        CommunicationTransportMode.GAS
    ),
    scenario: _CapturedLoopScenario | None = None,
) -> _ScenarioPreparedLoop:
    """Build a scenario loop and close its exact binding after setup failure."""
    cleanup_binding: list[tuple[Any, Any, Any]] = []
    try:
        return _build_scenario_prepared_loop_impl(
            monkeypatch,
            duration,
            cleanup_binding,
            device,
            communication_family,
            scenario,
        )
    except BaseException as setup_error:
        if cleanup_binding:
            session, registry, guard = cleanup_binding[0]
            try:
                session.close(registry, guard)
            except BaseException as cleanup_error:
                raise setup_error from cleanup_error
        raise


def _prepared_snapshot(loop: _PreparedLoop) -> dict[str, np.ndarray]:
    """Synchronize once and copy all meaningful resident-loop outputs."""
    loop.wp.synchronize_device(loop.session.particles.masses.device)
    particles = loop.session.particles
    gas = loop.session.gas
    environment = loop.session.environment
    snapshot = {
        "particle_masses": particles.masses.numpy().copy(),
        "particle_concentration": particles.concentration.numpy().copy(),
        "particle_charge": particles.charge.numpy().copy(),
        "particle_density": particles.density.numpy().copy(),
        "particle_volume": particles.volume.numpy().copy(),
        "gas_molar_mass": gas.molar_mass.numpy().copy(),
        "gas_concentration": gas.concentration.numpy().copy(),
        "gas_vapor_pressure": gas.vapor_pressure.numpy().copy(),
        "gas_partitioning": gas.partitioning.numpy().copy(),
        "temperature": environment.temperature.numpy().copy(),
        "pressure": environment.pressure.numpy().copy(),
        "saturation_ratio": environment.saturation_ratio.numpy().copy(),
        "collision_pairs": (
            loop.coagulation_resources.collision_pairs.numpy().copy()
        ),
        "collision_counts": (
            loop.coagulation_resources.n_collisions.numpy().copy()
        ),
        "coagulation_rng": (
            loop.coagulation_resources.rng_states.numpy().copy()
        ),
        "wall_loss_rng": loop.wall_loss_resources.rng_states.numpy().copy(),
    }
    for registration in loop.request.diagnostics.registrations:
        snapshot[f"diagnostic_{registration.operation.value}"] = (
            registration.output.numpy().copy()
        )
    return snapshot


def _scenario_snapshot(loop: _ScenarioPreparedLoop) -> dict[str, np.ndarray]:
    """Synchronize once at an assertion boundary and detach P1 outputs."""
    prepared = loop.loop
    prepared.wp.synchronize_device(prepared.session.particles.masses.device)
    snapshot = _prepared_snapshot_without_sync(prepared)
    buffers = loop.communication_resources.buffers
    work_buffers = (
        {
            "communication_amounts": buffers.amounts.numpy().copy(),
            "communication_deltas": buffers.amount_deltas.numpy().copy(),
            "communication_outbound": buffers.outbound_amounts.numpy().copy(),
        }
        if loop.communication_resources.configuration.communication_map.transport_mode
        is CommunicationTransportMode.GAS
        else {
            "communication_source_debits": buffers.source_debits.numpy().copy(),
            "communication_destination_credits": (
                buffers.destination_credits.numpy().copy()
            ),
            "communication_assignments": buffers.assignments.numpy().copy(),
            "communication_requests": (
                buffers.request_concentrations.numpy().copy()
            ),
        }
    )
    snapshot.update(
        work_buffers
        | {
            "latent_energy_input": loop.latent_energy.numpy().copy(),
            "baseline_total_mass": loop.baseline_total_mass.numpy().copy(),
            "source_ledger": loop.source_ledger.numpy().copy(),
            "sink_ledger": loop.sink_ledger.numpy().copy(),
        }
    )
    return snapshot


def _prepared_snapshot_without_sync(
    loop: _PreparedLoop,
) -> dict[str, np.ndarray]:
    """Copy prepared-loop arrays after the caller has synchronized once."""
    particles = loop.session.particles
    gas = loop.session.gas
    environment = loop.session.environment
    snapshot = {
        "particle_masses": particles.masses.numpy().copy(),
        "particle_concentration": particles.concentration.numpy().copy(),
        "particle_charge": particles.charge.numpy().copy(),
        "particle_density": particles.density.numpy().copy(),
        "particle_volume": particles.volume.numpy().copy(),
        "gas_molar_mass": gas.molar_mass.numpy().copy(),
        "gas_concentration": gas.concentration.numpy().copy(),
        "gas_vapor_pressure": gas.vapor_pressure.numpy().copy(),
        "gas_partitioning": gas.partitioning.numpy().copy(),
        "temperature": environment.temperature.numpy().copy(),
        "pressure": environment.pressure.numpy().copy(),
        "saturation_ratio": environment.saturation_ratio.numpy().copy(),
        "collision_pairs": (
            loop.coagulation_resources.collision_pairs.numpy().copy()
        ),
        "collision_counts": (
            loop.coagulation_resources.n_collisions.numpy().copy()
        ),
        "coagulation_rng": (
            loop.coagulation_resources.rng_states.numpy().copy()
        ),
        "wall_loss_rng": loop.wall_loss_resources.rng_states.numpy().copy(),
    }
    for registration in loop.request.diagnostics.registrations:
        snapshot[f"diagnostic_{registration.operation.value}"] = (
            registration.output.numpy().copy()
        )
    return snapshot


def _close_prepared_loop(loop: _PreparedLoop) -> None:
    """Release graph provenance before closing its resident session."""
    try:
        close_resident_graph_capture(loop.binding)
    except BaseException as graph_error:
        try:
            loop.session.close(loop.registry, loop.guard)
        except BaseException as cleanup_error:
            raise graph_error from cleanup_error
        raise
    loop.session.close(loop.registry, loop.guard)


@pytest.mark.warp
def test_scenario_builder_closes_session_when_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the exact resident binding if scenario setup cannot return it."""
    import particula.execution.gpu_session as gpu_session

    closed: list[tuple[Any, Any, Any]] = []

    def close(session: Any, registry: Any, guard: Any) -> None:
        closed.append((session, registry, guard))

    def fail_prepare(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("prepared setup failed")

    monkeypatch.setattr(gpu_session.ResidentSession, "close", close)
    monkeypatch.setattr(
        sys.modules[__name__], "prepare_resident_simulation", fail_prepare
    )
    with pytest.raises(RuntimeError, match="prepared setup failed"):
        _build_scenario_prepared_loop(monkeypatch, 0.0)
    assert len(closed) == 1


def test_close_prepared_loop_attempts_session_cleanup_after_graph_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attempt exact session cleanup while preserving graph-close failures."""
    closed: list[tuple[Any, Any]] = []
    loop = SimpleNamespace(
        binding=object(),
        session=SimpleNamespace(
            close=lambda registry, guard: closed.append((registry, guard))
        ),
        registry=object(),
        guard=object(),
    )

    def fail_graph_close(_binding: object) -> None:
        raise RuntimeError("graph close failed")

    monkeypatch.setattr(
        sys.modules[__name__], "close_resident_graph_capture", fail_graph_close
    )
    with pytest.raises(RuntimeError, match="graph close failed"):
        _close_prepared_loop(cast(_PreparedLoop, loop))
    assert closed == [(loop.registry, loop.guard)]


def _assert_prepared_parity(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> None:
    """Compare physical outputs without requiring cross-device RNG words."""
    ignored = {"coagulation_rng", "wall_loss_rng"}
    assert actual.keys() == expected.keys()
    for name in actual.keys() - ignored:
        if actual[name].dtype.kind in {"b", "i", "u"}:
            npt.assert_equal(actual[name], expected[name], err_msg=name)
        else:
            npt.assert_allclose(
                actual[name],
                expected[name],
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
                err_msg=name,
            )


def _snapshot_inventory(snapshot: dict[str, np.ndarray]) -> np.ndarray:
    """Compute per-box/species inventory from a detached snapshot."""
    return snapshot["particle_volume"][:, None] * (
        np.sum(
            snapshot["particle_masses"]
            * snapshot["particle_concentration"][:, :, None],
            axis=1,
        )
        + snapshot["gas_concentration"]
    )


def _prepared_identity_signature(loop: _PreparedLoop) -> tuple[int, ...]:
    """Return the identity-only resources retained by a prepared loop."""
    session = loop.session
    communication = loop.request.communication.resources
    diagnostics = loop.request.diagnostics.registrations
    return (
        id(session.particles),
        id(session.gas),
        id(session.environment),
        id(session.particles.masses),
        id(session.particles.concentration),
        id(session.particles.charge),
        id(session.particles.density),
        id(session.particles.volume),
        id(session.gas.molar_mass),
        id(session.gas.concentration),
        id(session.gas.vapor_pressure),
        id(session.gas.partitioning),
        id(session.environment.temperature),
        id(session.environment.pressure),
        id(session.environment.saturation_ratio),
        id(communication),
        *(id(registration.output) for registration in diagnostics),
    )


def _scenario_identity_signature(
    loop: _ScenarioPreparedLoop,
) -> tuple[int, ...]:
    """Return all primary, work, diagnostic, and accounting identities."""
    buffers = loop.communication_resources.buffers
    buffer_ids = (
        (
            id(buffers.amounts),
            id(buffers.amount_deltas),
            id(buffers.outbound_amounts),
        )
        if loop.communication_resources.configuration.communication_map.transport_mode
        is CommunicationTransportMode.GAS
        else (
            id(buffers.source_debits),
            id(buffers.destination_credits),
            id(buffers.assignments),
            id(buffers.request_concentrations),
        )
    )
    return _prepared_identity_signature(loop.loop) + (
        id(loop.communication_resources),
        *buffer_ids,
        id(loop.latent_energy),
        id(loop.baseline_total_mass),
        id(loop.source_ledger),
        id(loop.sink_ledger),
        id(loop.loop.prepared.resource_views),
    )


def _independent_gas_work_oracle(
    gas_concentration: np.ndarray,
    volume: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    enabled: np.ndarray,
    rates: np.ndarray,
    time_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate one GAS-communication work-buffer result without helpers."""
    amounts = gas_concentration * volume[:, None]
    deltas = np.zeros_like(amounts)
    outbound = np.zeros_like(amounts)
    for source, destination, is_enabled, rate in zip(
        sources, destinations, enabled, rates, strict=True
    ):
        if is_enabled and rate != 0.0:
            transfer = amounts[source] * rate * time_step
            deltas[source] -= transfer
            deltas[destination] += transfer
            outbound[source] += transfer
    return amounts, deltas, outbound


def _assert_dilution_inventory_factor(
    initial: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    coefficient: np.ndarray,
    time_step: float,
    steps: int,
) -> None:
    """Check detached per-box/species inventory against dilution alone."""
    npt.assert_allclose(
        _snapshot_inventory(result),
        _snapshot_inventory(initial)
        * np.exp(-coefficient[:, None] * time_step * steps),
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        err_msg="per-box/species dilution inventory",
    )


def _particle_transport_inventories(
    snapshot: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closed-map particle number, species-mass, and charge inventories."""
    amounts = (
        snapshot["particle_concentration"]
        * snapshot["particle_volume"][:, None]
    )
    number = np.array([np.sum(amounts)], dtype=np.float64)
    mass = np.sum(
        snapshot["particle_masses"] * amounts[:, :, None], axis=(0, 1)
    )
    charge = np.array(
        [np.sum(snapshot["particle_charge"] * amounts)], dtype=np.float64
    )
    return number, mass, charge


def _independent_particle_work_oracle(
    snapshot: dict[str, np.ndarray],
    scenario: _CapturedLoopScenario,
) -> dict[str, np.ndarray]:
    """Calculate detached PARTICLES state and planning buffers for one step."""
    concentration = snapshot["particle_concentration"].copy()
    masses = snapshot["particle_masses"].copy()
    charge = snapshot["particle_charge"].copy()
    boxes, slots = concentration.shape
    edges = len(scenario.edge_sources)
    debits = np.zeros_like(concentration)
    credits = np.zeros_like(concentration)
    assignments = np.full((edges, slots), -1, dtype=np.int32)
    requests = np.zeros((edges, slots), dtype=np.float64)
    for edge, (source, destination, enabled, rate) in enumerate(
        zip(
            scenario.edge_sources,
            scenario.edge_destinations,
            scenario.edge_enabled,
            scenario.edge_rates,
            strict=True,
        )
    ):
        if not enabled or rate == 0.0:
            continue
        for source_slot in range(slots):
            request = (
                concentration[source, source_slot] * rate * scenario.time_step
            )
            if request == 0.0:
                continue
            requests[edge, source_slot] = request
            debits[source, source_slot] += request
            target = next(
                (
                    target_slot
                    for target_slot in range(slots)
                    if concentration[destination, target_slot] > 0.0
                    and np.array_equal(
                        masses[source, source_slot],
                        masses[destination, target_slot],
                    )
                    and charge[source, source_slot]
                    == charge[destination, target_slot]
                ),
                None,
            )
            if target is None:
                target = next(
                    target_slot
                    for target_slot in range(slots)
                    if concentration[destination, target_slot] == 0.0
                    and target_slot not in assignments[:edge].ravel()
                    and target_slot not in assignments[edge, :source_slot]
                )
                masses[destination, target] = masses[source, source_slot]
                charge[destination, target] = charge[source, source_slot]
            assignments[edge, source_slot] = target
            credits[destination, target] += (
                request
                * snapshot["particle_volume"][source]
                / snapshot["particle_volume"][destination]
            )
    concentration += credits - debits
    for box in range(boxes):
        for slot in range(slots):
            if concentration[box, slot] == 0.0:
                masses[box, slot] = 0.0
                charge[box, slot] = 0.0
    concentration *= (
        snapshot["particle_volume"][:, None] / scenario.final_volume[:, None]
    )
    return {
        "particle_concentration": concentration,
        "particle_masses": masses,
        "particle_charge": charge,
        "communication_source_debits": debits,
        "communication_destination_credits": credits,
        "communication_assignments": assignments,
        "communication_requests": requests,
    }


def _assert_scenario_parity(
    actual: dict[str, np.ndarray],
    initial: dict[str, np.ndarray],
    scenario: _CapturedLoopScenario,
    steps: int,
) -> None:
    """Compare every P1 field and diagnostic to detached expectations."""
    expected = _run_captured_full_loop_oracle(scenario, steps)
    float_fields = {
        "particle_masses": expected.state.particle_masses,
        "particle_concentration": expected.state.particle_concentration,
        "particle_volume": expected.state.volume,
        "gas_concentration": expected.state.gas_concentration,
        "gas_vapor_pressure": expected.state.vapor_pressure,
        "saturation_ratio": expected.state.saturation_ratio,
        "diagnostic_gas_concentration_snapshot": expected.gas_concentration_snapshot,
        "diagnostic_saturation_ratio_snapshot": expected.saturation_ratio_snapshot,
        "diagnostic_total_species_mass": expected.total_species_mass,
        "diagnostic_particle_number_concentration": (
            expected.particle_number_concentration
        ),
        "diagnostic_latent_heat_energy": expected.latent_heat_energy,
        "diagnostic_conservation_residual": expected.conservation_residual,
    }
    for name, values in float_fields.items():
        npt.assert_allclose(
            actual[name],
            values,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg=name,
        )
    discrete_fields = {
        "particle_charge": scenario.particle_charge,
        "particle_density": np.array([1000.0, 1200.0]),
        "gas_molar_mass": scenario.molar_mass,
        "gas_partitioning": np.broadcast_to(
            scenario.gas_partitioning[0], (2, 2)
        ).astype(np.int32),
        "temperature": scenario.temperature,
        "pressure": scenario.pressure,
        "latent_energy_input": scenario.energy_ledger,
        "baseline_total_mass": scenario.baseline_total_mass,
        "source_ledger": scenario.source_ledger,
        "sink_ledger": scenario.sink_ledger,
    }
    for name, values in discrete_fields.items():
        if actual[name].dtype.kind in {"i", "u", "b"}:
            npt.assert_equal(actual[name], values, err_msg=name)
        else:
            npt.assert_allclose(
                actual[name],
                values,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
                err_msg=name,
            )
    # This fixture intentionally supplies no coagulation or wall-loss work.
    # Retained outputs and resident streams must therefore remain unchanged.
    for name in (
        "collision_pairs",
        "collision_counts",
        "coagulation_rng",
        "wall_loss_rng",
    ):
        npt.assert_equal(actual[name], initial[name], err_msg=name)


def _assert_scenario_conservation(
    snapshots: list[dict[str, np.ndarray]],
    scenario: _CapturedLoopScenario,
) -> None:
    """Check independent transport and retained dilution per timestep."""
    assert len(snapshots) == scenario.time_steps + 1
    for step, (before, after) in enumerate(
        zip(snapshots[:-1], snapshots[1:], strict=True), start=1
    ):
        _, deltas, _ = _independent_gas_work_oracle(
            before["gas_concentration"],
            before["particle_volume"],
            scenario.edge_sources,
            scenario.edge_destinations,
            scenario.edge_enabled,
            scenario.edge_rates,
            scenario.time_step,
        )
        transported = _snapshot_inventory(before) + deltas
        actual = _snapshot_inventory(after)
        npt.assert_allclose(
            actual
            / np.exp(
                -scenario.dilution_coefficient[:, None] * scenario.time_step
            ),
            transported,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg=f"transport and volume inventory before dilution at step {step}",
        )
        npt.assert_allclose(
            actual,
            transported
            * np.exp(
                -scenario.dilution_coefficient[:, None] * scenario.time_step
            ),
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg=f"retained dilution inventory at step {step}",
        )


def _deterministic_numpy_oracle(
    initial: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Advance the deterministic fixture without reading production output."""
    expected = {name: value.copy() for name, value in initial.items()}
    expected_vapor_pressure = np.full_like(initial["gas_vapor_pressure"], 800.0)
    expected_saturation = (
        initial["gas_concentration"]
        * GAS_CONSTANT
        * initial["temperature"][:, None]
        / (initial["gas_molar_mass"][None, :] * expected_vapor_pressure)
    )
    expected_inventory = _snapshot_inventory(initial)
    expected["gas_vapor_pressure"] = expected_vapor_pressure
    expected["saturation_ratio"] = expected_saturation
    expected["diagnostic_gas_concentration_snapshot"] = initial[
        "gas_concentration"
    ].copy()
    expected["diagnostic_saturation_ratio_snapshot"] = (
        expected_saturation.copy()
    )
    expected["diagnostic_total_species_mass"] = expected_inventory
    expected["diagnostic_particle_number_concentration"] = np.sum(
        initial["particle_concentration"], axis=1
    )
    expected["diagnostic_latent_heat_energy"] = np.zeros_like(
        initial["gas_concentration"]
    )
    # The fixture's independently allocated baseline/source/sink ledgers are
    # zero, so the diagnostic definition reduces to total species inventory.
    expected["diagnostic_conservation_residual"] = expected_inventory.copy()
    return expected


def _independent_wall_loss_sink(
    initial: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
) -> np.ndarray:
    """Measure removed initial particle inventory without differencing totals."""
    removed = (initial["particle_concentration"] > 0.0) & (
        result["particle_concentration"] == 0.0
    )
    removed_mass = np.sum(
        initial["particle_masses"]
        * initial["particle_concentration"][:, :, None]
        * removed[:, :, None],
        axis=1,
    )
    return initial["particle_volume"][:, None] * removed_mass


def _wall_loss_removal_moments(
    snapshot: dict[str, np.ndarray],
    duration: float,
    steps: int,
) -> tuple[float, float]:
    """Return independent Bernoulli mean and variance for active slots."""
    from particula.dynamics.properties.wall_loss_coefficient import (
        get_spherical_wall_loss_coefficient_via_system_state,
    )

    mean = 0.0
    variance = 0.0
    density = snapshot["particle_density"]
    for box in range(snapshot["particle_masses"].shape[0]):
        for slot in range(snapshot["particle_masses"].shape[1]):
            concentration = snapshot["particle_concentration"][box, slot]
            masses = snapshot["particle_masses"][box, slot]
            if concentration <= 0.0 or np.sum(masses) <= 0.0:
                continue
            particle_volume = np.sum(masses / density)
            radius = (3.0 * particle_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
            effective_density = float(np.sum(masses) / particle_volume)
            coefficient = float(
                get_spherical_wall_loss_coefficient_via_system_state(
                    wall_eddy_diffusivity=1.0,
                    particle_radius=radius,
                    particle_density=effective_density,
                    temperature=float(snapshot["temperature"][box]),
                    pressure=float(snapshot["pressure"][box]),
                    chamber_radius=1.0,
                )
            )
            probability = 1.0 - np.exp(-coefficient * duration * steps)
            mean += probability
            variance += probability * (1.0 - probability)
    return mean, variance


def _assert_same_state(
    actual: tuple[np.ndarray, ...], expected: tuple[np.ndarray, ...]
) -> None:
    """Compare float state tightly and integer metadata exactly."""
    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if index in {7, 12, 13, 14}:
            npt.assert_equal(actual_value, expected_value)
        else:
            npt.assert_allclose(
                actual_value,
                expected_value,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    (CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES),
)
def test_deterministic_full_loop_matches_independent_uncaptured_baseline(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
) -> None:
    """Three zero-duration steps preserve the full uncaptured resident state."""
    reference = _build_loop_fixture(monkeypatch, communication_family)
    initial_inventory = _inventory_oracle(reference)

    reference.scheduler.execute(0.0)
    stable_state = _snapshot(reference)
    for _ in range(2):
        reference.scheduler.execute(0.0)
        _assert_same_state(_snapshot(reference), stable_state)
        npt.assert_allclose(
            _inventory_oracle(reference),
            initial_inventory,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )

    expected_dispatch = tuple(
        node_id
        for node_id in reference.request.schedule.ordered_node_ids
        if node_id not in {"vapor_pressure_refresh", "saturation_refresh"}
    )
    assert reference.trace == list(expected_dispatch * 3)
    assert reference.guard.completed_steps == 3


def _native_cuda_candidates(wp: Any) -> tuple[Device, ...]:
    """Return normalized CUDA natives without replacement or fallback."""
    devices = (str(device) for device in wp.get_devices())
    return tuple(
        Device(Backend.WARP, native)
        for native in devices
        if native.startswith("cuda")
    )


def _require_native_cuda_capture() -> tuple[Any, tuple[Device, ...]]:
    """Skip only before native qualification when CUDA capture is unavailable."""
    wp = pytest.importorskip("warp")
    if not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)
    candidates = _native_cuda_candidates(wp)
    if not candidates:
        pytest.skip(CUDA_SKIP_REASON)
    if not all(
        callable(getattr(wp, name, None))
        for name in ("capture_begin", "capture_end", "capture_launch")
    ):
        pytest.skip("Warp capture API unavailable")
    return wp, candidates


def test_native_cuda_candidate_discovery_normalizes_warp_devices() -> None:
    """Discovery normalizes Warp-like objects and excludes non-CUDA values."""

    class DeviceValue:
        """Provide the string behavior of a Warp device object."""

        def __init__(self, native: str) -> None:
            self.native = native

        def __str__(self) -> str:
            return self.native

    first = "cuda:opaque:0"
    second = "cuda:opaque:1"
    fake_warp = SimpleNamespace(
        get_devices=lambda: (
            DeviceValue("cpu"),
            DeviceValue(first),
            object(),
            second,
            4,
        )
    )

    candidates = _native_cuda_candidates(fake_warp)

    assert tuple(candidate.native for candidate in candidates) == (
        first,
        second,
    )
    assert all(candidate.backend is Backend.WARP for candidate in candidates)


_EXPLICIT_CAPTURE_UNAVAILABILITY = frozenset(
    {
        "graph capture runtime is unavailable.",
        "graph capture device is unavailable.",
        "graph capture API is unsupported.",
    }
)


def _qualification_is_explicitly_unavailable(error: ValueError) -> bool:
    """Return whether qualification reported only an optional capability gap."""
    return str(error) in _EXPLICIT_CAPTURE_UNAVAILABILITY


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("graph capture runtime is unavailable.", True),
        ("graph capture device is unavailable.", True),
        ("graph capture API is unsupported.", True),
        ("prepared graph-capture identities do not match.", False),
        ("graph capture lifecycle must be ready.", False),
        ("prepared capture resource set does not match.", False),
    ),
)
def test_native_cuda_qualification_skips_only_explicit_unavailability(
    message: str, expected: bool
) -> None:
    """Keep binding, lifecycle, and resource failures fail closed."""
    assert (
        _qualification_is_explicitly_unavailable(ValueError(message))
        is expected
    )


def _forbid_prepared_host_work(
    dispatch_patch: pytest.MonkeyPatch,
    loop: _PreparedLoop,
    conversion: Any,
    forbidden: Any,
    forbid_resource_validation: bool = False,
) -> None:
    """Reject enqueue-time upload, allocation, copy, readback, and sync work."""
    for name in (
        "to_warp_particle_data",
        "to_warp_gas_data",
        "to_warp_environment_data",
    ):
        dispatch_patch.setattr(conversion, name, forbidden)
    # Patch module-level Warp aliases and the concrete resident-array readback
    # method used by the retained prepared operations.
    for name in (
        "synchronize",
        "synchronize_device",
        "zeros",
        "empty",
        "array",
        "copy",
        "to_numpy",
    ):
        if callable(getattr(loop.wp, name, None)):
            dispatch_patch.setattr(loop.wp, name, forbidden)
    array_type = type(loop.session.particles.masses)
    if callable(getattr(array_type, "numpy", None)):
        dispatch_patch.setattr(array_type, "numpy", forbidden)
    resource_methods = [
        "acquire_communication",
        "register_capture_resources",
        "prepare_capture_resources",
    ]
    if forbid_resource_validation:
        resource_methods.append("validate_capture_resource_set")
    for name in resource_methods:
        dispatch_patch.setattr(loop.registry, name, forbidden)


@pytest.mark.warp
def test_forbidden_host_work_detects_warp_array_method_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject the common resident Warp-array ``numpy`` readback boundary."""
    from particula.gpu import conversion

    loop = _build_prepared_loop("cpu", 1, 0.0, 1571)
    try:
        with monkeypatch.context() as replay_patch:

            def forbidden(*_args: object, **_kwargs: object) -> None:
                pytest.fail("native replay performed forbidden host work")

            _forbid_prepared_host_work(
                replay_patch,
                loop,
                conversion,
                forbidden,
            )
            with pytest.raises(
                pytest.fail.Exception,
                match="native replay performed forbidden host work",
            ):
                loop.session.particles.masses.numpy()
    finally:
        _close_prepared_loop(loop)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_real_uncaptured_warp_cpu_nonzero_loop_matches_numpy() -> None:
    """Exercise the native-test fixture on the required uncaptured baseline."""
    loop = _build_prepared_loop("cpu", 3, 0.25, 1571)
    try:
        initial = _prepared_snapshot(loop)
        identity_signature = _prepared_identity_signature(loop)
        expected = _deterministic_numpy_oracle(initial)
        for _ in range(3):
            enqueue_prepared_resident_simulation(loop.prepared)
        result = _prepared_snapshot(loop)

        _assert_prepared_parity(result, expected)
        npt.assert_allclose(
            _snapshot_inventory(result),
            _snapshot_inventory(initial),
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
        )
        assert loop.prepared.ordered_node_ids == (
            "communication",
            "volume_evolution",
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "condensation",
            "brownian_coagulation",
            "dilution",
            "wall_loss",
            "nucleation",
            "diagnostics",
        )
        assert tuple(
            operation.node.node_id for operation in loop.prepared.operations
        ) == (loop.prepared.ordered_node_ids)
        assert _prepared_identity_signature(loop) == identity_signature
        assert loop.guard.completed_steps == 3
    finally:
        _close_prepared_loop(loop)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_p1_ready_warp_cpu_loop_matches_detached_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise P1's real READY-only barriers without enqueue-time host work."""
    from particula.gpu import conversion

    scenario = _captured_full_loop_scenario()
    scenario_loop = _build_scenario_prepared_loop(
        monkeypatch, scenario.time_step
    )
    loop = scenario_loop.loop
    try:
        snapshots = [_scenario_snapshot(scenario_loop)]
        signature = _scenario_identity_signature(scenario_loop)
        assert loop.prepared.ordered_node_ids == (
            "communication",
            "volume_evolution",
            "environment_update",
            "gas_update",
            "vapor_pressure_refresh",
            "saturation_refresh",
            "condensation",
            "brownian_coagulation",
            "dilution",
            "wall_loss",
            "nucleation",
            "diagnostics",
        )
        assert tuple(
            node.node.node_id for node in loop.prepared.operations
        ) == (loop.prepared.ordered_node_ids)
        for _ in range(scenario.time_steps):
            with monkeypatch.context() as dispatch_patch:

                def forbidden(*_args: object, **_kwargs: object) -> None:
                    pytest.fail(
                        "prepared dispatch performed forbidden host work"
                    )

                _forbid_prepared_host_work(
                    dispatch_patch,
                    loop,
                    conversion,
                    forbidden,
                )
                enqueue_prepared_resident_simulation(loop.prepared)
            snapshots.append(_scenario_snapshot(scenario_loop))
        result = snapshots[-1]
        _assert_scenario_parity(
            result,
            snapshots[0],
            scenario,
            scenario.time_steps,
        )
        # Work buffers are overwritten for one dispatch, rather than accumulated.
        # The final snapshot is checked below against the independently evolved
        # pre-final state to keep this ledger oracle separate from scheduler code.
        pre_final = _run_captured_full_loop_oracle(
            scenario, scenario.time_steps - 1
        )
        amounts, deltas, outbound = _independent_gas_work_oracle(
            pre_final.state.gas_concentration,
            pre_final.state.volume,
            scenario.edge_sources,
            scenario.edge_destinations,
            scenario.edge_enabled,
            scenario.edge_rates,
            scenario.time_step,
        )
        npt.assert_allclose(
            result["communication_amounts"],
            amounts,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg="final communication amounts",
        )
        npt.assert_allclose(
            result["communication_deltas"],
            deltas,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg="final communication deltas",
        )
        npt.assert_allclose(
            result["communication_outbound"],
            outbound,
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            err_msg="final communication outbound",
        )
        _assert_scenario_conservation(
            snapshots,
            scenario,
        )
        assert _scenario_identity_signature(scenario_loop) == signature
        assert loop.guard.completed_steps == scenario.time_steps
        assert loop.binding.lifecycle.state is GraphCaptureLifecycleState.READY
    finally:
        _close_prepared_loop(loop)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_p1_ready_warp_cpu_zero_duration_is_write_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the exact P1 READY binding and payload state unchanged at zero time."""
    from particula.gpu import conversion

    scenario_loop = _build_scenario_prepared_loop(monkeypatch, 0.0)
    loop = scenario_loop.loop
    scenario = _captured_full_loop_scenario()
    try:
        before = _scenario_snapshot(scenario_loop)
        signature = _scenario_identity_signature(scenario_loop)
        with monkeypatch.context() as dispatch_patch:

            def forbidden(*_args: object, **_kwargs: object) -> None:
                pytest.fail(
                    "zero-duration prepared dispatch performed host work"
                )

            _forbid_prepared_host_work(
                dispatch_patch,
                loop,
                conversion,
                forbidden,
            )
            enqueue_prepared_resident_simulation(loop.prepared)
        after = _scenario_snapshot(scenario_loop)
        assert after.keys() == before.keys()
        refreshed = {
            "gas_vapor_pressure",
            "saturation_ratio",
            "diagnostic_gas_concentration_snapshot",
            "diagnostic_saturation_ratio_snapshot",
            "diagnostic_total_species_mass",
            "diagnostic_particle_number_concentration",
            "diagnostic_latent_heat_energy",
            "diagnostic_conservation_residual",
            "latent_energy_input",
        }
        for name in after.keys() - refreshed:
            npt.assert_equal(after[name], before[name], err_msg=name)
        expected_vapor_pressure = np.broadcast_to(
            scenario.vapor_pressure,
            scenario.gas_concentration.shape,
        )
        expected_saturation = (
            scenario.gas_concentration
            * _ORACLE_GAS_CONSTANT
            * scenario.temperature[:, None]
            / (scenario.molar_mass[None, :] * expected_vapor_pressure)
        )
        expected_inventory = _snapshot_inventory(before)
        npt.assert_equal(after["gas_vapor_pressure"], expected_vapor_pressure)
        npt.assert_allclose(after["saturation_ratio"], expected_saturation)
        npt.assert_equal(
            after["diagnostic_gas_concentration_snapshot"],
            before["gas_concentration"],
        )
        npt.assert_allclose(
            after["diagnostic_saturation_ratio_snapshot"], expected_saturation
        )
        npt.assert_allclose(
            after["diagnostic_total_species_mass"], expected_inventory
        )
        npt.assert_allclose(
            after["diagnostic_particle_number_concentration"],
            np.sum(before["particle_concentration"], axis=1),
        )
        npt.assert_equal(
            after["diagnostic_latent_heat_energy"], scenario.energy_ledger
        )
        npt.assert_equal(after["latent_energy_input"], scenario.energy_ledger)
        npt.assert_allclose(
            after["diagnostic_conservation_residual"],
            expected_inventory
            - scenario.baseline_total_mass
            - scenario.source_ledger
            + scenario.sink_ledger,
        )
        assert _scenario_identity_signature(scenario_loop) == signature
        assert loop.guard.completed_steps == 1
        assert loop.binding.lifecycle.state is GraphCaptureLifecycleState.READY
    finally:
        _close_prepared_loop(loop)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    (CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES),
)
def test_family_aware_ready_no_work_buffers_are_stable(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
) -> None:
    """Both closed-map families retain their owned work buffers at no work."""
    scenario_loop = _build_scenario_prepared_loop(
        monkeypatch,
        0.0,
        Device(Backend.WARP, "cpu"),
        communication_family,
    )
    loop = scenario_loop.loop
    try:
        before = _scenario_snapshot(scenario_loop)
        identities = _scenario_identity_signature(scenario_loop)
        enqueue_prepared_resident_simulation(loop.prepared)
        after = _scenario_snapshot(scenario_loop)
        work_names = tuple(
            name for name in before if name.startswith("communication_")
        )
        for name in work_names:
            npt.assert_equal(after[name], before[name], err_msg=name)
        assert _scenario_identity_signature(scenario_loop) == identities
        assert loop.guard.completed_steps == 1
        assert loop.binding.lifecycle.state is GraphCaptureLifecycleState.READY
    finally:
        _close_prepared_loop(loop)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    (CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES),
    ids=("gas", "particles"),
)
@pytest.mark.parametrize(
    "prescribed_volume", (False, True), ids=("fixed", "volume")
)
def test_family_aware_active_closed_map_preserves_detached_inventory(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
    prescribed_volume: bool,
) -> None:
    """Exercise active closed maps before optional volume evolution and dilution.

    The detached totals are intentionally checked independently from the
    scheduler's parity assertions.  With dilution disabled, a closed map and
    prescribed volume update preserve total particle number, species mass, and
    signed charge.  The GAS row additionally uses the existing detached gas
    oracle through ``_assert_scenario_parity``.
    """
    scenario = _captured_full_loop_scenario()
    scenario = replace(
        scenario,
        dilution_coefficient=_scenario_replacement([0.0, 0.0], np.float64),
        final_volume=(
            scenario.final_volume
            if prescribed_volume
            else _scenario_replacement(scenario.volume, np.float64)
        ),
    )
    scenario_loop = _build_scenario_prepared_loop(
        monkeypatch,
        scenario.time_step,
        Device(Backend.WARP, "cpu"),
        communication_family,
        scenario,
    )
    loop = scenario_loop.loop
    try:
        before = _scenario_snapshot(scenario_loop)
        identities = _scenario_identity_signature(scenario_loop)
        particle_before = _particle_transport_inventories(before)

        enqueue_prepared_resident_simulation(loop.prepared)

        after = _scenario_snapshot(scenario_loop)
        particle_after = _particle_transport_inventories(after)
        for actual, expected_inventory in zip(
            particle_after, particle_before, strict=True
        ):
            npt.assert_allclose(
                actual,
                expected_inventory,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )
        if communication_family is CommunicationTransportMode.GAS:
            _assert_scenario_parity(after, before, scenario, 1)
        else:
            expected = _independent_particle_work_oracle(before, scenario)
            for name, values in expected.items():
                if values.dtype.kind in {"i", "u", "b"}:
                    npt.assert_equal(after[name], values, err_msg=name)
                else:
                    npt.assert_allclose(
                        after[name],
                        values,
                        rtol=PARITY_RTOL,
                        atol=PARITY_ATOL,
                        err_msg=name,
                    )
            assert (
                after["particle_concentration"][0, 0]
                < before["particle_concentration"][0, 0]
            )
            assert np.any(after["particle_concentration"][1] > 0.0)
            assert np.any(after["communication_requests"] > 0.0)
        assert _scenario_identity_signature(scenario_loop) == identities
        assert loop.guard.completed_steps == 1
        assert loop.binding.lifecycle.state is GraphCaptureLifecycleState.READY
    finally:
        _close_prepared_loop(loop)


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_real_uncaptured_warp_cpu_zero_duration_preserves_primary_state() -> (
    None
):
    """A READY prepared zero-duration dispatch leaves primary state untouched."""
    loop = _build_prepared_loop("cpu", 3, 0.0, 1571)
    try:
        before = _prepared_snapshot(loop)
        identity_signature = _prepared_identity_signature(loop)
        enqueue_prepared_resident_simulation(loop.prepared)
        after = _prepared_snapshot(loop)

        for name in (
            "particle_masses",
            "particle_concentration",
            "particle_charge",
            "particle_density",
            "particle_volume",
            "gas_molar_mass",
            "gas_concentration",
            "gas_partitioning",
            "temperature",
            "pressure",
            "coagulation_rng",
            "wall_loss_rng",
        ):
            npt.assert_equal(after[name], before[name], err_msg=name)
        assert _prepared_identity_signature(loop) == identity_signature
        assert loop.guard.completed_steps == 1
        assert loop.binding.lifecycle.state is GraphCaptureLifecycleState.READY
    finally:
        _close_prepared_loop(loop)


def _capture_scenario(kind: str) -> _CapturedLoopScenario:
    """Return an immutable active, volume, or write-free capture scenario."""
    scenario = _captured_full_loop_scenario()
    if kind == "active":
        return scenario
    if kind == "volume":
        return replace(
            scenario,
            dilution_coefficient=_scenario_replacement([0.0, 0.0], np.float64),
        )
    if kind == "no-work":
        return replace(
            scenario,
            edge_enabled=_scenario_replacement([False, False, False], np.bool_),
            edge_rates=_scenario_replacement([0.0, 0.0, 0.0], np.float64),
            dilution_coefficient=_scenario_replacement([0.0, 0.0], np.float64),
            final_volume=_scenario_replacement(scenario.volume, np.float64),
        )
    raise ValueError(f"unknown capture scenario: {kind}")


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    (CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES),
    ids=("gas", "particles"),
)
@pytest.mark.parametrize("scenario_kind", ("active", "volume", "no-work"))
def test_native_cuda_family_capture_replay_matches_uncaptured_baseline(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
    scenario_kind: str,
) -> None:
    """Replay each native CUDA candidate without host work or CPU fallback."""
    from particula.gpu import conversion

    wp, candidates = _require_native_cuda_capture()
    scenario = _capture_scenario(scenario_kind)
    qualified_candidates = 0
    for cuda_device in candidates:
        cpu_loop = _build_scenario_prepared_loop(
            monkeypatch,
            scenario.time_step,
            Device(Backend.WARP, "cpu"),
            communication_family,
            scenario,
        )
        cuda_loop: _ScenarioPreparedLoop | None = None
        try:
            cuda_loop = _build_scenario_prepared_loop(
                monkeypatch,
                scenario.time_step,
                cuda_device,
                communication_family,
                scenario,
            )
            cpu_before = _scenario_snapshot(cpu_loop)
            cuda_before = _scenario_snapshot(cuda_loop)
            _assert_same_state(
                tuple(cuda_before.values()), tuple(cpu_before.values())
            )
            capture_set = cuda_loop.loop.registry.validate_capture_resource_set(
                cuda_loop.loop.request.capture_resource_requirements
            )
            try:
                qualification = qualify_prepared_resident_graph_capture(
                    cuda_loop.loop.binding,
                    cuda_loop.loop.prepared,
                    capture_set,
                    _WarpNativeCaptureAdapter(wp, cuda_device.native),
                )
            except ValueError as error:
                if _qualification_is_explicitly_unavailable(error):
                    continue
                raise
            qualified_candidates += 1
            captured = capture_prepared_resident_graph(qualification)
            with monkeypatch.context() as replay_patch:

                def forbidden(*_args: object, **_kwargs: object) -> None:
                    pytest.fail("native replay performed forbidden host work")

                _forbid_prepared_host_work(
                    replay_patch,
                    cuda_loop.loop,
                    conversion,
                    forbidden,
                )
                replay_captured_resident_graph(captured, qualification.duration)
            enqueue_prepared_resident_simulation(cpu_loop.loop.prepared)
            cuda_after = _scenario_snapshot(cuda_loop)
            cpu_after = _scenario_snapshot(cpu_loop)
            _assert_same_state(
                tuple(cuda_after.values()), tuple(cpu_after.values())
            )
            assert cuda_loop.loop.guard.completed_steps == 1
            assert (
                cuda_loop.loop.binding.lifecycle.state
                is GraphCaptureLifecycleState.CAPTURED
            )
            assert not hasattr(captured, "handle")
        finally:
            if cuda_loop is not None:
                _close_prepared_loop(cuda_loop.loop)
            _close_prepared_loop(cpu_loop.loop)
    if not qualified_candidates:
        pytest.skip("no CUDA candidate qualified for native capture")


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.parametrize("n_boxes", (1, 3))
def test_native_cuda_nonzero_full_loop_matches_numpy_and_uncaptured_warp(
    n_boxes: int,
) -> None:
    """Replay every qualified nonzero CUDA graph against Warp CPU and NumPy."""
    wp, candidates = _require_native_cuda_capture()
    duration = 0.25
    steps = 3
    qualified_candidates = 0
    for cuda_device in candidates:
        cpu_loop = _build_prepared_loop("cpu", n_boxes, duration, 1571)
        cuda_loop = None
        try:
            cuda_loop = _build_prepared_loop(
                cuda_device.native, n_boxes, duration, 1571
            )
            cpu_initial = _prepared_snapshot(cpu_loop)
            cuda_initial = _prepared_snapshot(cuda_loop)
            _assert_prepared_parity(cuda_initial, cpu_initial)
            expected = _deterministic_numpy_oracle(cpu_initial)
            initial_inventory = _snapshot_inventory(cpu_initial)
            capture_set = cuda_loop.registry.validate_capture_resource_set(
                cuda_loop.request.capture_resource_requirements
            )
            try:
                qualification = qualify_prepared_resident_graph_capture(
                    cuda_loop.binding,
                    cuda_loop.prepared,
                    capture_set,
                    _WarpNativeCaptureAdapter(wp, cuda_device.native),
                )
            except ValueError as error:
                if _qualification_is_explicitly_unavailable(error):
                    continue
                raise
            qualified_candidates += 1
            captured = capture_prepared_resident_graph(qualification)
            _assert_prepared_parity(_prepared_snapshot(cuda_loop), cuda_initial)
            for _ in range(steps):
                enqueue_prepared_resident_simulation(cpu_loop.prepared)
                replay_captured_resident_graph(captured, qualification.duration)
            cpu_result = _prepared_snapshot(cpu_loop)
            cuda_result = _prepared_snapshot(cuda_loop)
            _assert_prepared_parity(cuda_result, cpu_result)
            _assert_prepared_parity(cpu_result, expected)
            _assert_prepared_parity(cuda_result, expected)
            npt.assert_allclose(
                _snapshot_inventory(cuda_result),
                initial_inventory,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )
            npt.assert_equal(cuda_result["gas_partitioning"], 0)
            assert cpu_loop.guard.completed_steps == steps
            assert cuda_loop.guard.completed_steps == steps
            assert not hasattr(captured, "handle")
        finally:
            if cuda_loop is not None:
                _close_prepared_loop(cuda_loop)
            _close_prepared_loop(cpu_loop)
    if not qualified_candidates:
        pytest.skip("no CUDA candidate qualified for native capture")


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.stochastic
def test_native_cuda_wall_loss_rng_and_stochastic_aggregate() -> None:
    """Compare isolated wall-loss aggregates without coagulation clearing."""
    wp, candidates = _require_native_cuda_capture()
    duration = 1.0
    steps = 2
    seeds = range(12)
    cpu_removed = 0
    cuda_removed = 0
    expected_removed = 0.0
    expected_variance = 0.0
    continued_words = 0

    for seed in seeds:
        selected = (0, 1, 2)
        cpu_loop = _build_prepared_loop(
            "cpu",
            len(selected),
            duration,
            seed,
            selected_wall_loss_boxes=selected,
            single_active_particle=True,
        )
        cuda_loop = None
        captured = None
        try:
            cuda_loop = _build_prepared_loop(
                candidates[0].native,
                len(selected),
                duration,
                seed,
                selected_wall_loss_boxes=selected,
                single_active_particle=True,
            )
            cpu_initial = _prepared_snapshot(cpu_loop)
            cuda_initial = _prepared_snapshot(cuda_loop)
            _assert_prepared_parity(cuda_initial, cpu_initial)
            mean, variance = _wall_loss_removal_moments(
                cpu_initial,
                duration,
                steps,
            )
            expected_removed += mean
            expected_variance += variance

            capture_set = cuda_loop.registry.validate_capture_resource_set(
                cuda_loop.request.capture_resource_requirements
            )
            try:
                qualification = qualify_prepared_resident_graph_capture(
                    cuda_loop.binding,
                    cuda_loop.prepared,
                    capture_set,
                    _WarpNativeCaptureAdapter(
                        wp,
                        cuda_loop.session.metadata.device.native,
                    ),
                )
            except ValueError as error:
                if _qualification_is_explicitly_unavailable(error):
                    pytest.skip(
                        "no CUDA candidate qualified for native capture: "
                        f"{error}"
                    )
                raise
            captured = capture_prepared_resident_graph(qualification)
            assert (
                cuda_loop.registry.coagulation_resources.rng_states
                is not cuda_loop.registry.wall_loss_resources.rng_states
            )
            cuda_rng_initial = cuda_initial["wall_loss_rng"]

            enqueue_prepared_resident_simulation(cpu_loop.prepared)
            replay_captured_resident_graph(captured, qualification.duration)
            cuda_first = _prepared_snapshot(cuda_loop)
            assert np.any(cuda_first["wall_loss_rng"] != cuda_rng_initial)

            enqueue_prepared_resident_simulation(cpu_loop.prepared)
            replay_captured_resident_graph(captured, qualification.duration)
            cpu_result = _prepared_snapshot(cpu_loop)
            cuda_result = _prepared_snapshot(cuda_loop)
            continued_words += int(
                np.count_nonzero(
                    cuda_result["wall_loss_rng"] != cuda_first["wall_loss_rng"]
                )
            )

            cpu_removed += int(
                np.count_nonzero(
                    (cpu_initial["particle_concentration"] > 0.0)
                    & (cpu_result["particle_concentration"] == 0.0)
                )
            )
            cuda_removed += int(
                np.count_nonzero(
                    (cuda_initial["particle_concentration"] > 0.0)
                    & (cuda_result["particle_concentration"] == 0.0)
                )
            )
            for initial, result in (
                (cpu_initial, cpu_result),
                (cuda_initial, cuda_result),
            ):
                initial_inventory = _snapshot_inventory(initial)
                final_inventory = _snapshot_inventory(result)
                sink_inventory = _independent_wall_loss_sink(initial, result)
                assert np.all(final_inventory <= initial_inventory)
                npt.assert_allclose(
                    final_inventory + sink_inventory,
                    initial_inventory,
                    rtol=PARITY_RTOL,
                    atol=PARITY_ATOL,
                )
        finally:
            captured = None
            if cuda_loop is not None:
                _close_prepared_loop(cuda_loop)
            _close_prepared_loop(cpu_loop)

    sigma = np.sqrt(expected_variance)
    single_path_bound = max(4.0 * sigma, 2.0)
    cross_path_bound = max(4.0 * np.sqrt(2.0) * sigma, 2.0)
    assert abs(cpu_removed - expected_removed) <= single_path_bound
    assert abs(cuda_removed - expected_removed) <= single_path_bound
    assert abs(cuda_removed - cpu_removed) <= cross_path_bound
    assert continued_words > 0


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.stochastic
def test_native_cuda_coagulation_rng_and_nonvacuous_activity() -> None:
    """Compare isolated Brownian activity with an independent opportunity gate."""
    wp, candidates = _require_native_cuda_capture()
    duration = 1.0
    steps = 2
    seeds = range(12)
    cpu_collisions = 0
    cuda_collisions = 0
    collision_opportunities = 0
    continued_words = 0

    for seed in seeds:
        cpu_loop = _build_prepared_loop("cpu", 3, duration, seed)
        cuda_loop = None
        try:
            cuda_loop = _build_prepared_loop(
                candidates[0].native,
                3,
                duration,
                seed,
            )
            cpu_initial = _prepared_snapshot(cpu_loop)
            cuda_initial = _prepared_snapshot(cuda_loop)
            _assert_prepared_parity(cuda_initial, cpu_initial)
            active_per_box = np.count_nonzero(
                cpu_initial["particle_concentration"] > 0.0,
                axis=1,
            )
            collision_opportunities += int(
                steps * np.sum(active_per_box * (active_per_box - 1) // 2)
            )
            capture_set = cuda_loop.registry.validate_capture_resource_set(
                cuda_loop.request.capture_resource_requirements
            )
            try:
                qualification = qualify_prepared_resident_graph_capture(
                    cuda_loop.binding,
                    cuda_loop.prepared,
                    capture_set,
                    _WarpNativeCaptureAdapter(
                        wp,
                        cuda_loop.session.metadata.device.native,
                    ),
                )
            except ValueError as error:
                if _qualification_is_explicitly_unavailable(error):
                    pytest.skip(
                        "no CUDA candidate qualified for native capture: "
                        f"{error}"
                    )
                raise
            captured = capture_prepared_resident_graph(qualification)
            first_words = cuda_initial["coagulation_rng"]
            for step in range(steps):
                enqueue_prepared_resident_simulation(cpu_loop.prepared)
                replay_captured_resident_graph(captured, qualification.duration)
                if step == 0:
                    cuda_first = _prepared_snapshot(cuda_loop)
                    assert np.any(cuda_first["coagulation_rng"] != first_words)
            cpu_result = _prepared_snapshot(cpu_loop)
            cuda_result = _prepared_snapshot(cuda_loop)
            continued_words += int(
                np.count_nonzero(
                    cuda_result["coagulation_rng"]
                    != cuda_first["coagulation_rng"]
                )
            )
            cpu_collisions += int(np.sum(cpu_result["collision_counts"]))
            cuda_collisions += int(np.sum(cuda_result["collision_counts"]))
            npt.assert_allclose(
                _snapshot_inventory(cpu_result),
                _snapshot_inventory(cpu_initial),
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )
            npt.assert_allclose(
                _snapshot_inventory(cuda_result),
                _snapshot_inventory(cuda_initial),
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )
        finally:
            if cuda_loop is not None:
                _close_prepared_loop(cuda_loop)
            _close_prepared_loop(cpu_loop)

    assert collision_opportunities > 0
    assert cpu_collisions > 0
    assert cuda_collisions > 0
    collision_bound = max(4.0 * np.sqrt(cpu_collisions + cuda_collisions), 2.0)
    assert abs(cuda_collisions - cpu_collisions) <= collision_bound
    assert continued_words > 0


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_capture_support_is_native_only() -> None:
    """Native CUDA is an optional capture boundary, never a Warp-CPU fallback."""
    _wp, candidates = _require_native_cuda_capture()
    assert candidates


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_capture_zero_duration_contract_is_available_before_capture() -> (
    None
):
    """CUDA availability is decided before a zero-duration capture is attempted."""
    wp, _candidates = _require_native_cuda_capture()
    assert callable(wp.capture_begin)
    assert callable(wp.capture_end)


@pytest.mark.warp
@pytest.mark.gpu_parity
@pytest.mark.parametrize(
    "communication_family",
    (CommunicationTransportMode.GAS, CommunicationTransportMode.PARTICLES),
)
def test_fake_native_capture_replays_nonzero_reference_steps_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    communication_family: CommunicationTransportMode,
) -> None:
    """Capture and replay preserve a deterministic multi-step reference state."""
    fixture = _build_loop_fixture(monkeypatch, communication_family)
    qualification, binding, native = _prepared_fake_capture(
        fixture, monkeypatch, 1.25
    )
    import particula.execution.resident_scheduler as resident_scheduler

    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_captured_prepared_operations",
        lambda _prepared: None,
    )

    try:
        before = _snapshot(fixture)
        captured = capture_prepared_resident_graph(qualification)
        after_capture = _snapshot(fixture)
        replay_captured_resident_graph(captured, qualification.duration)
        after_replay = _snapshot(fixture)

        _assert_same_state(after_capture, before)
        _assert_same_state(after_replay, before)
        replay_captured_resident_graph(captured, qualification.duration)
        _assert_same_state(_snapshot(fixture), before)
        assert not hasattr(captured, "handle")
        assert native.calls == [
            "runtime",
            "device:cuda:0",
            "api:cuda:0",
            "callables:cuda:0",
            "begin",
            "end",
            "launch",
            "launch",
        ]
        assert native.launches == [native.handle, native.handle]
        assert binding.lifecycle.state is GraphCaptureLifecycleState.CAPTURED
        assert fixture.guard.completed_steps == 2
    finally:
        close_resident_graph_capture(binding)
    assert native.calls[-1] == "release"
    assert native.calls.count("release") == 1


@pytest.mark.warp
def test_qualification_rejection_skips_capture_before_guard_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row-local native rejection cannot begin capture or open a step token."""
    fixture = _build_loop_fixture(monkeypatch, CommunicationTransportMode.GAS)
    qualification, binding, _native = _prepared_fake_capture(
        fixture, monkeypatch, 1.0
    )

    class RejectedAdapter:
        """Reject runtime qualification before native callable resolution."""

        def runtime_available(self) -> bool:
            return False

        def device_available(self, device: Device) -> bool:
            raise AssertionError(f"rejected adapter resolved device {device}")

        def capture_api_available(self, device: Device) -> bool:
            raise AssertionError(f"rejected adapter resolved API {device}")

        def capture_callables(
            self, device: Device
        ) -> GraphCaptureNativeCallables:
            raise AssertionError(
                f"rejected adapter resolved callables {device}"
            )

    capture_set = fixture.registry.validate_capture_resource_set(
        fixture.request.capture_resource_requirements
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            sys.modules[__name__],
            "capture_prepared_resident_graph",
            lambda _qualification: pytest.fail("rejected row captured"),
        )
        with pytest.raises(ValueError, match="runtime"):
            qualify_prepared_resident_graph_capture(
                binding,
                qualification.prepared,
                capture_set,
                RejectedAdapter(),
            )
    assert fixture.guard.completed_steps == 0
    close_resident_graph_capture(binding)
    assert binding.lifecycle.state is GraphCaptureLifecycleState.CLOSED


@pytest.mark.warp
def test_captured_replay_rejects_stale_binding_and_duration_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duration and terminal binding drift are rejected without native launch."""
    fixture = _build_loop_fixture(monkeypatch, CommunicationTransportMode.GAS)
    qualification, binding, native = _prepared_fake_capture(
        fixture, monkeypatch, 2.0
    )
    import particula.execution.resident_scheduler as resident_scheduler

    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_captured_prepared_operations",
        lambda _prepared: None,
    )
    captured = capture_prepared_resident_graph(qualification)

    try:
        with pytest.raises(ValueError, match="duration"):
            replay_captured_resident_graph(captured, 1.0)
        assert native.launches == []
        assert fixture.guard.completed_steps == 0
    finally:
        close_resident_graph_capture(binding)
    with pytest.raises(ValueError, match="not P2-issued"):
        replay_captured_resident_graph(captured, qualification.duration)
    assert native.launches == []
    assert native.calls.count("release") == 1


@pytest.mark.warp
def test_capture_replay_has_no_forbidden_host_work_and_preserves_rng_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fake native capture proves no transfer, sync, instantiate, or RNG reset."""
    fixture = _build_loop_fixture(monkeypatch, CommunicationTransportMode.GAS)
    qualification, binding, native = _prepared_fake_capture(
        fixture, monkeypatch, 0.0
    )
    import particula.execution.resident_scheduler as resident_scheduler
    from particula.gpu import conversion

    monkeypatch.setattr(
        resident_scheduler,
        "_enqueue_captured_prepared_operations",
        lambda _prepared: None,
    )
    monkeypatch.setattr(
        fixture.wp,
        "synchronize",
        lambda: pytest.fail("capture/replay must not synchronize"),
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_particle_data",
        lambda *_args, **_kwargs: pytest.fail(
            "capture/replay must not transfer"
        ),
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_gas_data",
        lambda *_args, **_kwargs: pytest.fail(
            "capture/replay must not transfer"
        ),
    )
    monkeypatch.setattr(
        conversion,
        "to_warp_environment_data",
        lambda *_args, **_kwargs: pytest.fail(
            "capture/replay must not transfer"
        ),
    )
    coagulation_rng = (
        fixture.request.coagulation.resources.rng_states.numpy().copy()
    )
    wall_loss_rng = (
        fixture.request.wall_loss.resources.rng_states.numpy().copy()
    )
    captured = capture_prepared_resident_graph(qualification)
    try:
        replay_captured_resident_graph(captured, qualification.duration)

        assert "instantiate" not in native.calls
        npt.assert_equal(
            fixture.request.coagulation.resources.rng_states.numpy(),
            coagulation_rng,
        )
        npt.assert_equal(
            fixture.request.wall_loss.resources.rng_states.numpy(),
            wall_loss_rng,
        )
        assert fixture.guard.completed_steps == 1
    finally:
        close_resident_graph_capture(binding)
