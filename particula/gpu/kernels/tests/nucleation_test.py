"""Focused read-only contract tests for direct GPU nucleation P1."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.warp
wp = pytest.importorskip("warp")

from particula.gpu.kernels.nucleation import (  # noqa: E402
    NucleationConfig,
    NucleationDiagnosticBuffers,
    NucleationFinalizedDemandBuffers,
    NucleationScratchBuffers,
    _preflight_nucleation,
    _real,
)


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
    before = scratch.potential_rate.numpy().copy()
    result = _preflight_nucleation(
        particles,
        gas,
        _config(),
        1.0,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    assert result.n_boxes == 1
    assert result.n_particles == 2
    np.testing.assert_array_equal(scratch.potential_rate.numpy(), before)


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
    before = scratch.potential_demand.numpy().copy()
    result = _preflight_nucleation(
        particles,
        gas,
        _config(**changes),
        time_step,
        temperature=300.0,
        scratch=scratch,
        finalized_demand=finalized,
        diagnostics=diagnostics,
    )
    assert result.gate_reason == reason
    assert result.potential_rate == 0.0
    np.testing.assert_array_equal(scratch.potential_demand.numpy(), before)


def test_rejection_does_not_write_state():
    """Schema/science rejection occurs before caller state is modified."""
    particles, gas = _state()
    before = particles.concentration.numpy().copy()
    with pytest.raises(ValueError, match="temperature"):
        _preflight_nucleation(particles, gas, _config(), 1.0, temperature=100.0)
    np.testing.assert_array_equal(particles.concentration.numpy(), before)


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
