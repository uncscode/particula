"""Partial P1 three-way evidence boundaries for resident graph-capture loops.

Warp CPU supplies the uncaptured resident baseline. Native CUDA capture is
optional and is skipped before qualification when the device or Warp capture
API is unavailable; it is never emulated on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from particula.execution import Backend, Device
from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationMap,
    CommunicationMapForm,
    CommunicationResourceShape,
    CommunicationShapeKind,
    CommunicationTransportMode,
    PrescribedVolumeUpdate,
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
from particula.execution.process_adapters import ResidentNucleationRequest
from particula.execution.resident_scheduler import (
    enqueue_prepared_resident_simulation,
    prepare_resident_simulation,
)
from particula.execution.tests.full_loop_test import _build_loop_fixture
from particula.execution.tests.multi_box_loop_test import (
    _binding,
    _resident_graph,
    _scheduler_request,
)
from particula.gpu.kernels.nucleation import (
    NucleationConfig,
    NucleationExhaustionControls,
)
from particula.gpu.tests.cuda_availability import (
    CUDA_SKIP_REASON,
    cuda_available,
)
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
) -> _PreparedLoop:
    """Build one real all-operation resident loop on the requested device."""
    manifest = tuple((f"box-{2 * index}", index) for index in range(n_boxes))
    session, registry, guard = _binding(device, manifest, root_seed)
    wp = pytest.importorskip("warp")
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


def _close_prepared_loop(loop: _PreparedLoop) -> None:
    """Release graph provenance before closing its resident session."""
    close_resident_graph_capture(loop.binding)
    loop.session.close(loop.registry, loop.guard)


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


def _require_native_cuda_capture() -> Any:
    """Skip only before native qualification when CUDA capture is unavailable."""
    wp = pytest.importorskip("warp")
    if not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)
    if not any(str(device).startswith("cuda") for device in wp.get_devices()):
        pytest.skip(CUDA_SKIP_REASON)
    if not all(
        callable(getattr(wp, name, None))
        for name in ("capture_begin", "capture_end", "capture_launch")
    ):
        pytest.skip("Warp capture API unavailable")
    return wp


@pytest.mark.warp
@pytest.mark.gpu_parity
def test_real_uncaptured_warp_cpu_nonzero_loop_matches_numpy() -> None:
    """Exercise the native-test fixture on the required uncaptured baseline."""
    loop = _build_prepared_loop("cpu", 3, 0.25, 1571)
    try:
        initial = _prepared_snapshot(loop)
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
        assert loop.guard.completed_steps == 3
    finally:
        _close_prepared_loop(loop)


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.parametrize("n_boxes", (1, 3))
def test_native_cuda_nonzero_full_loop_matches_numpy_and_uncaptured_warp(
    n_boxes: int,
) -> None:
    """Replay a genuine nonzero CUDA graph against Warp CPU and NumPy."""
    wp = _require_native_cuda_capture()
    duration = 0.25
    steps = 3
    cpu_loop = _build_prepared_loop("cpu", n_boxes, duration, 1571)
    cuda_loop = None
    captured = None
    try:
        cuda_loop = _build_prepared_loop("cuda", n_boxes, duration, 1571)
        cpu_initial = _prepared_snapshot(cpu_loop)
        cuda_initial = _prepared_snapshot(cuda_loop)
        _assert_prepared_parity(cuda_initial, cpu_initial)
        expected = _deterministic_numpy_oracle(cpu_initial)
        initial_inventory = _snapshot_inventory(cpu_initial)

        capture_set = cuda_loop.registry.validate_capture_resource_set(
            cuda_loop.request.capture_resource_requirements
        )
        adapter = _WarpNativeCaptureAdapter(
            wp,
            cuda_loop.session.metadata.device.native,
        )
        qualification = qualify_prepared_resident_graph_capture(
            cuda_loop.binding,
            cuda_loop.prepared,
            capture_set,
            adapter,
        )
        captured = capture_prepared_resident_graph(qualification)
        # Native recording itself is not a hidden physical launch.
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
        captured = None
        if cuda_loop is not None:
            _close_prepared_loop(cuda_loop)
        _close_prepared_loop(cpu_loop)


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
@pytest.mark.stochastic
def test_native_cuda_rng_continuation_and_stochastic_aggregate() -> None:
    """Compare wall-loss aggregates without cross-device trajectories."""
    wp = _require_native_cuda_capture()
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
        )
        cuda_loop = None
        captured = None
        try:
            cuda_loop = _build_prepared_loop(
                "cuda",
                len(selected),
                duration,
                seed,
                selected_wall_loss_boxes=selected,
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
            qualification = qualify_prepared_resident_graph_capture(
                cuda_loop.binding,
                cuda_loop.prepared,
                capture_set,
                _WarpNativeCaptureAdapter(
                    wp,
                    cuda_loop.session.metadata.device.native,
                ),
            )
            captured = capture_prepared_resident_graph(qualification)
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
def test_cuda_capture_support_is_native_only() -> None:
    """Native CUDA is an optional capture boundary, never a Warp-CPU fallback."""
    wp = _require_native_cuda_capture()
    devices = [
        device for device in wp.get_devices() if str(device).startswith("cuda")
    ]
    assert devices


@pytest.mark.warp
@pytest.mark.cuda
@pytest.mark.gpu_parity
def test_cuda_capture_zero_duration_contract_is_available_before_capture() -> (
    None
):
    """CUDA availability is decided before a zero-duration capture is attempted."""
    wp = _require_native_cuda_capture()
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
