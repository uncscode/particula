"""Focused read-only contract tests for direct GPU nucleation P1."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.warp
wp = pytest.importorskip("warp")

from particula.gpu.kernels.nucleation import (  # noqa: E402
    _P2_GATE_ELIGIBLE,
    _P2_GATE_GAS_LIMITED_OFFSET,
    _P2_GATE_LOW_SATURATION,
    _P2_GATE_ZERO_COEFFICIENT,
    _P2_GATE_ZERO_INVENTORY,
    _P2_GATE_ZERO_PRECURSOR,
    _P2_GATE_ZERO_SURVIVAL,
    _P2_GATE_ZERO_TIME,
    NucleationConfig,
    NucleationDiagnosticBuffers,
    NucleationFinalizedDemandBuffers,
    NucleationScratchBuffers,
    _plan_nucleation_demand,
    _preflight_nucleation,
    _real,
)
from particula.util.constants import AVOGADRO_NUMBER  # noqa: E402


def _state(boxes: int = 1, particles: int = 2, species: int = 1):
    """Build valid caller-owned fixed-capacity state on Warp CPU."""
    particle_data = SimpleNamespace(
        masses=wp.zeros(
            (boxes, particles, species), dtype=wp.float64, device="cpu"
        ),
        concentration=wp.ones(
            (boxes, particles), dtype=wp.float64, device="cpu"
        ),
        charge=wp.zeros((boxes, particles), dtype=wp.float64, device="cpu"),
        density=wp.ones(species, dtype=wp.float64, device="cpu")
        * wp.float64(1000.0),
        volume=wp.ones(boxes, dtype=wp.float64, device="cpu"),
    )
    gas_data = SimpleNamespace(
        molar_mass=wp.ones(species, dtype=wp.float64, device="cpu")
        * wp.float64(0.1),
        concentration=wp.ones((boxes, species), dtype=wp.float64, device="cpu"),
        partitioning=wp.ones((boxes, species), dtype=wp.int32, device="cpu"),
    )
    return particle_data, gas_data


def _config(**changes):
    """Build a valid activation configuration."""
    values = dict(
        rate_law="activation",
        coefficient=1.0,
        survival_factor=1.0,
        precursor_index=0,
        molecule_counts=(1,),
        formation_diameter=1e-9,
        precursor_number_concentration_lower=0.0,
        precursor_number_concentration_upper=1e30,
        temperature_lower=200.0,
        temperature_upper=400.0,
    )
    values.update(changes)
    return NucleationConfig(**values)


def _sidecars(boxes: int, particles: int, species: int):
    """Build all future caller-owned sidecars with deliberately stale values."""
    return (
        NucleationScratchBuffers(
            *[wp.ones(boxes, dtype=wp.float64, device="cpu") for _ in range(3)]
        ),
        NucleationFinalizedDemandBuffers(
            wp.ones(boxes, dtype=wp.int32, device="cpu"),
            wp.ones(boxes, dtype=wp.float64, device="cpu"),
            wp.ones((boxes, species), dtype=wp.float64, device="cpu"),
        ),
        NucleationDiagnosticBuffers(
            wp.ones(boxes, dtype=wp.int32, device="cpu"),
            wp.ones((boxes, particles), dtype=wp.int32, device="cpu"),
        ),
    )


def _snapshot_arrays(*owners):
    """Capture caller-owned Warp array identity, schema, and exact contents."""
    snapshot = []
    for owner in owners:
        for value in owner.__dict__.values():
            if hasattr(value, "numpy"):
                snapshot.append(
                    (
                        value,
                        tuple(value.shape),
                        value.dtype,
                        str(value.device),
                        value.numpy().copy(),
                    )
                )
    return snapshot


def _assert_snapshot_unchanged(snapshot):
    """Assert a complete caller-owned array snapshot remains unchanged."""
    for value, shape, dtype, device, contents in snapshot:
        assert tuple(value.shape) == shape
        assert value.dtype == dtype
        assert str(value.device) == device
        np.testing.assert_array_equal(value.numpy(), contents, strict=True)


def test_config_is_frozen_but_retains_arrays_by_identity():
    """Frozen records reject rebinding without copying caller array sidecars."""
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.coefficient = 2.0
    array = wp.zeros(1, dtype=wp.float64, device="cpu")
    buffers = NucleationScratchBuffers(array, array, array)
    assert buffers.precursor_number_concentration is array


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rate_law": "other"}, "rate_law"),
        ({"coefficient": True}, "coefficient"),
        ({"formation_diameter": 0.0}, "formation_diameter"),
        ({"molecule_counts": (0,)}, "molecule_counts"),
    ],
)
def test_config_rejects_invalid_scalar_rules(changes, message):
    """Configuration validation rejects invalid scalar-only contracts."""
    with pytest.raises((TypeError, ValueError), match=message):
        _config(**changes)


@pytest.mark.parametrize(
    ("value", "positive", "exception"),
    [
        (True, False, TypeError),
        ("1.0", False, TypeError),
        (np.nan, False, ValueError),
        (-1.0, False, ValueError),
        (0.0, True, ValueError),
    ],
)
def test_real_rejects_nonphysical_scalar_values(value, positive, exception):
    """Scalar normalization rejects booleans, non-reals, and bad domains."""
    with pytest.raises(exception):
        _real(value, "value", positive=positive)


def test_config_accepts_kinetic_saturation_bounds():
    """A complete kinetic configuration retains valid paired bounds."""
    config = _config(
        rate_law="kinetic",
        saturation_lower=0.1,
        saturation_upper=1.2,
    )
    assert config.rate_law == "kinetic"
    assert config.saturation_upper == 1.2


@pytest.mark.parametrize(
    "changes",
    [
        {"precursor_index": True},
        {"molecule_counts": [1]},
        {"molecule_counts": (1.0,)},
        {"temperature_lower": 400.0, "temperature_upper": 200.0},
        {"saturation_lower": 0.1},
        {"saturation_lower": 1.0, "saturation_upper": 0.1},
    ],
)
def test_config_rejects_inconsistent_metadata_and_intervals(changes):
    """Configuration rejects malformed species metadata and paired intervals."""
    with pytest.raises((TypeError, ValueError)):
        _config(**changes)


def test_preflight_is_read_only_for_valid_inputs():
    """Valid P1 preflight retains state and returns normalized metadata."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    environment = SimpleNamespace(
        temperature=wp.full(
            1, wp.float64(300.0), dtype=wp.float64, device="cpu"
        ),
    )
    before = _snapshot_arrays(
        particles, gas, environment, scratch, finalized, diagnostics
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        environment=environment,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    assert result.n_boxes == 1
    assert result.n_particles == 2
    _assert_snapshot_unchanged(before)


def test_preflight_accepts_finite_signed_particle_charge():
    """Finite signed elementary-charge state remains valid P1 input."""
    particles, gas = _state()
    particles.charge = wp.array(
        np.array([[-1.0, 1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )

    result = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
    )

    assert result.has_eligible_boxes


def test_preflight_accepts_environment_temperature_and_saturation():
    """Environment-owned arrays support configured saturation gating metadata."""
    particles, gas = _state()
    environment = SimpleNamespace(
        temperature=wp.ones(1, dtype=wp.float64, device="cpu")
        * wp.float64(300.0),
        saturation_ratio=wp.ones((1, 1), dtype=wp.float64, device="cpu"),
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        environment=environment,
    )
    assert result.temperature is environment.temperature
    assert result.saturation is environment.saturation_ratio
    assert result.has_eligible_boxes


def test_preflight_empty_capacity_returns_no_eligible_boxes():
    """Empty fixed-capacity storage is valid but has no eligible work."""
    particles, gas = _state(particles=0)
    result = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
    )
    assert result.n_particles == 0
    assert not result.has_eligible_boxes
    assert not result.has_gated_boxes


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", 100.0, "temperature"),
        ("temperature", 500.0, "temperature"),
    ],
)
def test_preflight_rejects_scalar_temperature_outside_bounds(
    field, value, message
):
    """Scalar environmental values must remain inside configured bounds."""
    particles, gas = _state()
    with pytest.raises(ValueError, match=message):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            **{field: value},
        )


@pytest.mark.parametrize(
    ("temperature", "saturation", "message"),
    [
        (100.0, 1.0, "temperature"),
        (500.0, 1.0, "temperature"),
        (300.0, 1.6, "saturation"),
    ],
)
def test_preflight_rejects_direct_array_values_outside_bounds(
    temperature, saturation, message
):
    """Device-resident temperature and saturation arrays enforce bounds."""
    particles, gas = _state()
    temperature_data = wp.ones(1, dtype=wp.float64, device="cpu") * wp.float64(
        temperature
    )
    saturation_data = wp.ones(
        (1, 1), dtype=wp.float64, device="cpu"
    ) * wp.float64(saturation)
    before = temperature_data.numpy().copy()
    with pytest.raises(ValueError, match=message):
        _preflight_nucleation(
            particles,
            gas,
            _config(saturation_lower=0.5, saturation_upper=1.5),
            1.0,
            temperature=temperature_data,
            saturation=saturation_data,
        )
    np.testing.assert_array_equal(temperature_data.numpy(), before)


def test_preflight_low_saturation_is_a_read_only_gate():
    """Saturation below its lower endpoint gates rather than rejects."""
    particles, gas = _state()
    saturation_data = wp.full(
        (1, 1), wp.float64(0.4), dtype=wp.float64, device="cpu"
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        temperature=wp.full(
            1, wp.float64(300.0), dtype=wp.float64, device="cpu"
        ),
        saturation=saturation_data,
    )
    assert result.gate_reason == "low_saturation"


@pytest.mark.parametrize(
    ("time_step", "changes", "reason"),
    [
        (0.0, {}, "zero_time"),
        (1.0, {"coefficient": 0.0}, "zero_coefficient"),
        (1.0, {"survival_factor": 0.0}, "zero_survival"),
    ],
)
def test_zero_gates_leave_stale_sidecars_untouched(time_step, changes, reason):
    """All-box scalar gates return zero metadata without clearing sidecars."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    temperature = wp.full(1, wp.float64(300.0), dtype=wp.float64, device="cpu")
    before = _snapshot_arrays(
        particles, gas, temperature, scratch, finalized, diagnostics
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(**changes),
        time_step,
        temperature=temperature,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    assert result.gate_reason == reason
    assert result.potential_rate == 0.0
    _assert_snapshot_unchanged(before)


def test_rejection_does_not_write_state():
    """Schema/science rejection occurs before caller state is modified."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    temperature = wp.full(1, wp.float64(100.0), dtype=wp.float64, device="cpu")
    before = _snapshot_arrays(
        particles, gas, temperature, scratch, finalized, diagnostics
    )
    with pytest.raises(ValueError, match="temperature"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=temperature,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )
    _assert_snapshot_unchanged(before)


@pytest.mark.parametrize(
    ("configure", "message"),
    [
        (
            lambda particles, gas: setattr(
                gas,
                "partitioning",
                wp.zeros((1, 1), dtype=wp.int32, device="cpu"),
            ),
            "precursor partitioning",
        ),
        (
            lambda particles, gas: setattr(
                gas,
                "partitioning",
                wp.ones((1, 1), dtype=wp.int32, device="cpu") * 2,
            ),
            "binary",
        ),
    ],
)
def test_preflight_rejects_invalid_precursor_partitioning(configure, message):
    """Precursor partitioning must be enabled and use binary metadata."""
    particles, gas = _state()
    configure(particles, gas)
    with pytest.raises(ValueError, match=message):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
        )


def test_preflight_rejects_conflicting_or_unconfigured_environment_inputs():
    """Direct and environment forms cannot be mixed or silently ignored."""
    particles, gas = _state()
    environment = SimpleNamespace(
        temperature=wp.ones(1, dtype=wp.float64, device="cpu")
        * wp.float64(300.0),
    )
    with pytest.raises(ValueError, match="require no environment"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            environment=environment,
        )
    with pytest.raises(ValueError, match="not configured"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            saturation=wp.ones((1, 1), dtype=wp.float64, device="cpu"),
        )


def test_preflight_rejects_sidecar_overlap_without_mutating_state():
    """Sidecars cannot alias mutable particle fields during a read-only call."""
    particles, gas = _state()
    before = particles.concentration.numpy().copy()
    scratch = NucleationScratchBuffers(
        particles.volume,
        wp.zeros(1, dtype=wp.float64, device="cpu"),
        wp.zeros(1, dtype=wp.float64, device="cpu"),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        _preflight_nucleation(
            particles,
            gas,
            _config(),
            1.0,
            temperature=300.0,
            scratch=scratch,
        )
    np.testing.assert_array_equal(particles.concentration.numpy(), before)


def test_real_rejects_oversized_scalars_as_value_errors():
    """Scalar conversion overflow follows the finite-domain error contract."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _real(10**1000, "value")


def test_config_and_time_step_reject_oversized_scalars_as_value_errors():
    """Configuration and direct scalar overflow do not leak OverflowError."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _config(coefficient=10**1000)
    particles, gas = _state()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _preflight_nucleation(
            particles, gas, _config(), 10**1000, temperature=300.0
        )


def test_preflight_rejects_lookalike_and_mutated_config_without_writes():
    """Runtime config validation rejects untrusted indices before kernel use."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)
    before = _snapshot_arrays(particles, gas, scratch, finalized, diagnostics)
    lookalike = SimpleNamespace(**_config().__dict__)
    with pytest.raises(ValueError, match="NucleationConfig"):
        _preflight_nucleation(
            particles,
            gas,
            lookalike,
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )
    config = _config()
    object.__setattr__(config, "precursor_index", -1)
    with pytest.raises(ValueError, match="config scalar values"):
        _preflight_nucleation(
            particles,
            gas,
            config,
            1.0,
            temperature=300.0,
            scratch=scratch,
            finalized_demand=finalized,
            diagnostics=diagnostics,
        )
    _assert_snapshot_unchanged(before)


@pytest.mark.parametrize("temperature", [100.0, 500.0])
@pytest.mark.parametrize(
    "changes", [{"coefficient": 0.0}, {"survival_factor": 0.0}]
)
def test_scalar_zero_config_gates_before_temperature_interval(
    temperature, changes
):
    """Valid zero rate inputs short-circuit configured scalar temperature bounds."""
    particles, gas = _state()
    result = _preflight_nucleation(
        particles, gas, _config(**changes), 1.0, temperature=temperature
    )
    assert result.gate_reason in ("zero_coefficient", "zero_survival")


@pytest.mark.parametrize("temperature", [100.0, 500.0])
@pytest.mark.parametrize(
    "changes", [{"coefficient": 0.0}, {"survival_factor": 0.0}]
)
def test_device_zero_config_gates_before_temperature_interval(
    temperature, changes
):
    """Valid zero rate inputs short-circuit device temperature range checks."""
    particles, gas = _state()
    temperature_data = wp.full(
        1, wp.float64(temperature), dtype=wp.float64, device="cpu"
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(**changes),
        1.0,
        temperature=temperature_data,
    )
    assert result.gate_reason in ("zero_coefficient", "zero_survival")


def test_disjoint_precursor_and_saturation_gates_are_all_box_union():
    """Disjoint per-box gates do not incorrectly leave an eligible box."""
    particles, gas = _state(boxes=2)
    gas.concentration = wp.array(
        np.array([[0.0], [1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    saturation = wp.array(
        np.array([[1.0], [0.4]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    result = _preflight_nucleation(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        temperature=wp.full(
            2, wp.float64(300.0), dtype=wp.float64, device="cpu"
        ),
        saturation=saturation,
    )
    assert not result.has_eligible_boxes
    assert result.has_gated_boxes


def test_mixed_gate_union_retains_eligible_box():
    """A gate union smaller than box count retains eligible work metadata."""
    particles, gas = _state(boxes=2)
    gas.concentration = wp.array(
        np.array([[0.0], [1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    saturation = wp.ones((2, 1), dtype=wp.float64, device="cpu")
    result = _preflight_nucleation(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        temperature=wp.full(
            2, wp.float64(300.0), dtype=wp.float64, device="cpu"
        ),
        saturation=saturation,
    )
    assert result.has_eligible_boxes
    assert result.has_gated_boxes


def test_p2_plans_activation_demand_and_preserves_p3_sidecars():
    """P2 commits only demand diagnostics using shared inventory admission."""
    particles, gas = _state(boxes=2, particles=0, species=2)
    gas.molar_mass = wp.array(
        np.array([0.1, 0.2], dtype=np.float64), dtype=wp.float64, device="cpu"
    )
    gas.concentration = wp.array(
        np.array([[1.0, 0.1], [2.0, 2.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(2, 0, 2)
    source_before = _snapshot_arrays(particles, gas)
    selected_before = diagnostics.selected_slot_indices.numpy().copy()
    counts_before = finalized.accepted_counts.numpy().copy()
    config = _config(molecule_counts=(1, 1), coefficient=1.0e-10)

    _plan_nucleation_demand(
        particles,
        gas,
        config,
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    precursor = gas.concentration.numpy()[:, 0] * 6.02214076e23 / 0.1
    potential = 1.0e-10 * precursor
    event_mass = np.array([0.1, 0.2]) / 6.02214076e23
    expected = np.minimum(
        potential,
        np.min(gas.concentration.numpy() / event_mass, axis=1),
    )
    np.testing.assert_allclose(
        scratch.precursor_number_concentration.numpy(), precursor, rtol=1e-12
    )
    np.testing.assert_allclose(
        scratch.potential_rate.numpy(), potential, rtol=1e-12
    )
    np.testing.assert_allclose(
        scratch.potential_demand.numpy(), potential, rtol=1e-12
    )
    np.testing.assert_allclose(
        finalized.accepted_demand.numpy(), expected, rtol=1e-12
    )
    np.testing.assert_allclose(
        finalized.precursor_mass_change.numpy(),
        expected[:, None] * event_mass,
        rtol=1e-12,
    )
    assert diagnostics.gate_codes.numpy()[0] in (
        _P2_GATE_ELIGIBLE,
        _P2_GATE_GAS_LIMITED_OFFSET,
        _P2_GATE_GAS_LIMITED_OFFSET + 1,
    )
    np.testing.assert_array_equal(
        finalized.accepted_counts.numpy(), counts_before
    )
    np.testing.assert_array_equal(
        diagnostics.selected_slot_indices.numpy(), selected_before
    )
    _assert_snapshot_unchanged(source_before)


def test_p2_zero_time_writes_zero_outputs_and_gate_code():
    """P2 scalar gates commit their defined zero diagnostics."""
    particles, gas = _state()
    scratch, finalized, diagnostics = _sidecars(1, 2, 1)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(),
        0.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(scratch.potential_rate.numpy(), [0.0])
    np.testing.assert_array_equal(scratch.potential_demand.numpy(), [0.0])
    np.testing.assert_array_equal(finalized.accepted_demand.numpy(), [0.0])
    np.testing.assert_array_equal(
        finalized.precursor_mass_change.numpy(), [[0.0]]
    )
    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), [_P2_GATE_ZERO_TIME]
    )


@pytest.mark.parametrize(
    ("changes", "concentration", "saturation_value", "expected_code"),
    [
        ({"coefficient": 0.0}, 1.0, 1.0, _P2_GATE_ZERO_COEFFICIENT),
        ({"survival_factor": 0.0}, 1.0, 1.0, _P2_GATE_ZERO_SURVIVAL),
        ({}, 0.0, 1.0, _P2_GATE_ZERO_PRECURSOR),
        ({}, 1.0, 0.4, _P2_GATE_LOW_SATURATION),
        ({}, 1.0, 1.0, _P2_GATE_ZERO_INVENTORY),
    ],
)
def test_p2_writes_all_non_time_gate_codes(
    changes, concentration, saturation_value, expected_code
):
    """P2 commits zero demand for each valid gate with defined precedence."""
    particles, gas = _state()
    gas.concentration = wp.full(
        (1, 1), wp.float64(concentration), dtype=wp.float64, device="cpu"
    )
    if expected_code == _P2_GATE_ZERO_INVENTORY:
        gas.concentration = wp.zeros((1, 1), dtype=wp.float64, device="cpu")
        changes = {"molecule_counts": (0, 1)}
        particles, gas = _state(species=2)
        gas.concentration = wp.array(
            np.array([[1.0, 0.0]], dtype=np.float64),
            dtype=wp.float64,
            device="cpu",
        )
    boxes, particle_count, species = 1, 2, gas.molar_mass.shape[0]
    scratch, finalized, diagnostics = _sidecars(boxes, particle_count, species)
    config = _config(
        **changes,
        saturation_lower=0.5,
        saturation_upper=1.5,
    )
    saturation = wp.full(
        (1, species),
        wp.float64(saturation_value),
        dtype=wp.float64,
        device="cpu",
    )

    _plan_nucleation_demand(
        particles,
        gas,
        config,
        1.0,
        temperature=300.0,
        saturation=saturation,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), [expected_code]
    )
    if expected_code == _P2_GATE_ZERO_INVENTORY:
        assert scratch.potential_rate.numpy()[0] > 0.0
    else:
        np.testing.assert_array_equal(scratch.potential_rate.numpy(), [0.0])
    np.testing.assert_array_equal(finalized.accepted_demand.numpy(), [0.0])
    np.testing.assert_array_equal(
        finalized.precursor_mass_change.numpy(), np.zeros((1, species))
    )


def test_p2_kinetic_tied_inventory_uses_lowest_limiting_species():
    """Kinetic P2 demand uses its squared rate law and stable limiter tie."""
    particles, gas = _state(species=2)
    gas.molar_mass = wp.array(
        np.array([0.1, 0.2], dtype=np.float64), dtype=wp.float64, device="cpu"
    )
    event_mass = np.array([0.1, 0.2], dtype=np.float64) / 6.02214076e23
    gas.concentration = wp.array(
        np.array([event_mass * 5.0], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(1, 2, 2)
    config = _config(
        rate_law="kinetic",
        coefficient=1.0,
        molecule_counts=(1, 1),
        precursor_number_concentration_upper=1e40,
    )

    _plan_nucleation_demand(
        particles,
        gas,
        config,
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    precursor = gas.concentration.numpy()[0, 0] * 6.02214076e23 / 0.1
    np.testing.assert_allclose(scratch.potential_rate.numpy(), [precursor**2])
    np.testing.assert_allclose(finalized.accepted_demand.numpy(), [5.0])
    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(), [_P2_GATE_GAS_LIMITED_OFFSET]
    )


def test_p2_gate_precedence_is_scalar_then_precursor_then_saturation():
    """P2 emits the documented deterministic gate precedence per box."""
    particles, gas = _state(boxes=2)
    gas.concentration = wp.array(
        np.array([[0.0], [1.0]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    saturation = wp.array(
        np.array([[0.4], [0.4]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(2, 2, 1)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        0.0,
        temperature=300.0,
        saturation=saturation,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(),
        [_P2_GATE_ZERO_TIME, _P2_GATE_ZERO_TIME],
    )

    _plan_nucleation_demand(
        particles,
        gas,
        _config(saturation_lower=0.5, saturation_upper=1.5),
        1.0,
        temperature=300.0,
        saturation=saturation,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_array_equal(
        diagnostics.gate_codes.numpy(),
        [_P2_GATE_ZERO_PRECURSOR, _P2_GATE_LOW_SATURATION],
    )


def test_p2_shared_admission_preserves_nonparticipant_zero_removal():
    """P2 limits each box independently and never removes nonparticipants."""
    particles, gas = _state(boxes=2, particles=0, species=3)
    molar_mass = np.array([0.1, 0.3, 0.2], dtype=np.float64)
    event_mass = np.array([0.1, 0.0, 0.4], dtype=np.float64) / AVOGADRO_NUMBER
    gas.molar_mass = wp.array(molar_mass, dtype=wp.float64, device="cpu")
    gas.concentration = wp.array(
        np.array(
            [
                [1.0, 9.0, event_mass[2] * 2.0],
                [0.5, 8.0, event_mass[2] * 7.0],
            ],
            dtype=np.float64,
        ),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(2, 0, 3)
    before = _snapshot_arrays(particles, gas)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(
            coefficient=1.0e-10,
            molecule_counts=(1, 0, 2),
            precursor_number_concentration_upper=1.0e30,
        ),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    potential = (
        1.0e-10 * gas.concentration.numpy()[:, 0] * AVOGADRO_NUMBER / 0.1
    )
    expected_demand = np.minimum(potential, [2.0, 7.0])
    expected_removal = expected_demand[:, None] * event_mass
    np.testing.assert_allclose(
        finalized.accepted_demand.numpy(), expected_demand, rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        finalized.precursor_mass_change.numpy(),
        expected_removal,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        finalized.precursor_mass_change.numpy()[:, 1], [0.0, 0.0]
    )
    _assert_snapshot_unchanged(before)


def test_p2_empty_boxes_are_an_exact_write_free_noop():
    """Valid empty-box planning preserves every supplied sidecar byte-for-byte."""
    particles, gas = _state(boxes=0)
    scratch, finalized, diagnostics = _sidecars(0, 2, 1)
    before = _snapshot_arrays(particles, gas, scratch, finalized, diagnostics)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    _assert_snapshot_unchanged(before)


def test_p2_inventory_rounding_uses_bounded_nextafter_correction():
    """P2 corrects an unsafe division-product ULP without a coarse scale loss."""
    event_mass = np.float64(0.1 / AVOGADRO_NUMBER)
    inventory = None
    expected_demand = None
    for events in range(1, 4097):
        for direction in (np.inf, -np.inf):
            candidate_inventory = np.nextafter(
                event_mass * np.float64(events), direction
            )
            candidate_demand = candidate_inventory / event_mass
            if candidate_demand * event_mass > candidate_inventory:
                inventory = candidate_inventory
                expected_demand = candidate_demand
                break
        if inventory is not None:
            break
    assert inventory is not None
    assert expected_demand is not None

    for _ in range(4):
        if expected_demand * event_mass <= inventory:
            break
        expected_demand = np.nextafter(expected_demand, -np.inf)
    assert expected_demand * event_mass <= inventory

    particles, gas = _state(particles=0)
    gas.concentration = wp.array(
        np.array([[inventory]], dtype=np.float64),
        dtype=wp.float64,
        device="cpu",
    )
    scratch, finalized, diagnostics = _sidecars(1, 0, 1)

    _plan_nucleation_demand(
        particles,
        gas,
        _config(coefficient=1.0e10),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )

    np.testing.assert_allclose(
        finalized.accepted_demand.numpy(), [expected_demand], rtol=0.0, atol=0.0
    )
    assert finalized.precursor_mass_change.numpy()[0, 0] <= inventory
