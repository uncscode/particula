"""Numerical-contract tests for chamber dilution helpers."""

import numpy as np
import numpy.testing as npt
import particula.dynamics as dynamics
import particula.dynamics.dilution as dilution_module
import pytest
from particula.aerosol import Aerosol
from particula.dynamics.dilution import (
    dilute_aerosol,
    get_dilution_rate,
    get_dilution_step,
    get_volume_dilution_coefficient,
)
from particula.gas import EnvironmentData, GasData
from particula.particles.particle_data import ParticleData


@pytest.mark.parametrize(
    "volume, flow_rate",
    [
        (10.0, 2.0),
        (np.array([10.0, 20.0]), np.array([2.0, 8.0])),
        (np.array([[10.0], [20.0]]), np.array([2.0, 8.0])),
    ],
)
def test_volume_dilution_coefficient_equation_and_broadcasting(
    volume, flow_rate
):
    """Coefficient follows Q / V with scalar and array return conventions."""
    result = get_volume_dilution_coefficient(volume, flow_rate)
    expected = np.asarray(flow_rate, dtype=np.float64) / np.asarray(
        volume,
        dtype=np.float64,
    )

    npt.assert_allclose(result, expected)
    if np.ndim(volume) == 0 and np.ndim(flow_rate) == 0:
        assert np.isscalar(result)
    else:
        assert isinstance(result, np.ndarray)
        assert result.shape == expected.shape


@pytest.mark.parametrize(
    "coefficient, concentration",
    [
        (0.5, 10.0),
        (np.array([0.5, 1.0]), np.array([10.0, 4.0])),
        (np.array([[0.5], [1.0]]), np.array([10.0, 4.0])),
    ],
)
def test_dilution_rate_equation_and_broadcasting(coefficient, concentration):
    """Instantaneous rate follows -alpha c with broadcast output shape."""
    result = get_dilution_rate(coefficient, concentration)
    expected = -np.asarray(coefficient, dtype=np.float64) * np.asarray(
        concentration,
        dtype=np.float64,
    )

    npt.assert_allclose(result, expected)
    if np.ndim(coefficient) == 0 and np.ndim(concentration) == 0:
        assert np.isscalar(result)
    else:
        assert isinstance(result, np.ndarray)
        assert result.shape == expected.shape


@pytest.mark.parametrize(
    "coefficient, concentration, time_step",
    [
        (0.5, 10.0, 2.0),
        (np.array([0.5, 1.0]), np.array([10.0, 4.0]), 2.0),
        (np.array([[0.5], [1.0]]), np.array([10.0, 4.0]), [1.0, 2.0]),
    ],
)
def test_dilution_step_equation_and_broadcasting(
    coefficient, concentration, time_step
):
    """Finite update follows the independent exact exponential expression."""
    result = get_dilution_step(coefficient, concentration, time_step)
    expected = np.asarray(concentration, dtype=np.float64) * np.exp(
        -np.asarray(coefficient, dtype=np.float64)
        * np.asarray(time_step, dtype=np.float64)
    )

    npt.assert_allclose(result, expected)
    if all(
        np.ndim(value) == 0 for value in (coefficient, concentration, time_step)
    ):
        assert np.isscalar(result)
    else:
        assert isinstance(result, np.ndarray)
        assert result.shape == expected.shape


def test_dilution_step_extreme_finite_decay_is_warning_clean():
    """Finite coefficient-time overflow produces exact, finite zero decay."""
    result = get_dilution_step(
        np.finfo(np.float64).max,
        1.0,
        np.finfo(np.float64).max,
    )

    assert result == 0.0
    assert np.isfinite(result)
    assert result >= 0.0


@pytest.mark.parametrize(
    ("function", "arguments", "expected"),
    [
        (
            get_volume_dilution_coefficient,
            (np.array(10.0), 2.0),
            np.array(0.2),
        ),
        (get_dilution_rate, (np.array(0.5), 10.0), np.array(-5.0)),
        (get_dilution_step, (0.5, np.array(10.0), 2.0), np.array(10.0 / np.e)),
    ],
)
def test_dilution_helpers_preserve_zero_dimensional_array_results(
    function, arguments, expected
):
    """A zero-dimensional ndarray input produces a zero-dimensional ndarray."""
    result = function(*arguments)

    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    npt.assert_allclose(result, expected)


def test_dilution_rate_uint64_inputs_do_not_wrap_positive():
    """Unsigned inputs are converted before signed rate arithmetic."""
    coefficient = np.uint64(2**63)
    concentration = np.uint64(2)

    result = get_dilution_rate(coefficient, concentration)
    expected = -np.float64(coefficient) * np.float64(concentration)

    assert result <= 0.0
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("function", "arguments", "quantity"),
    [
        (
            get_volume_dilution_coefficient,
            (np.finfo(np.float64).tiny, np.finfo(np.float64).max),
            "dilution coefficient",
        ),
        (
            get_dilution_rate,
            (np.finfo(np.float64).max, np.finfo(np.float64).max),
            "dilution rate",
        ),
    ],
)
def test_dilution_helpers_reject_overflowed_results_warning_cleanly(
    function, arguments, quantity
):
    """Overflowed helper results fail clearly without emitting warnings."""
    with pytest.raises(ValueError, match=f"Computed {quantity} must be finite"):
        function(*arguments)


def test_dilution_step_uint64_inputs_decay_without_overflow():
    """Unsigned inputs are converted before exponential step arithmetic."""
    coefficient = np.uint64(2**63)
    concentration = np.uint64(3)
    time_step = np.uint64(2)

    result = get_dilution_step(coefficient, concentration, time_step)
    expected = np.float64(concentration) * np.exp(
        -np.float64(coefficient) * np.float64(time_step)
    )

    assert np.isfinite(result)
    assert 0.0 <= result <= concentration
    npt.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        (get_volume_dilution_coefficient, (0.0, 1.0), "must be positive"),
        (get_volume_dilution_coefficient, (-1.0, 1.0), "must be positive"),
        (get_volume_dilution_coefficient, (np.nan, 1.0), "must be finite"),
        (get_volume_dilution_coefficient, (np.inf, 1.0), "must be finite"),
        (get_volume_dilution_coefficient, (1.0, -1.0), "must be nonnegative"),
        (get_volume_dilution_coefficient, (1.0, np.nan), "must be finite"),
        (get_volume_dilution_coefficient, (1.0, np.inf), "must be finite"),
        (get_dilution_rate, (-1.0, 1.0), "must be nonnegative"),
        (get_dilution_rate, (np.nan, 1.0), "must be finite"),
        (get_dilution_rate, (np.inf, 1.0), "must be finite"),
        (get_dilution_rate, (1.0, -1.0), "must be nonnegative"),
        (get_dilution_rate, (1.0, np.nan), "must be finite"),
        (get_dilution_rate, (1.0, np.inf), "must be finite"),
        (get_dilution_step, (-1.0, 1.0, 1.0), "must be nonnegative"),
        (get_dilution_step, (np.nan, 1.0, 1.0), "must be finite"),
        (get_dilution_step, (np.inf, 1.0, 1.0), "must be finite"),
        (get_dilution_step, (1.0, -1.0, 1.0), "must be nonnegative"),
        (get_dilution_step, (1.0, np.nan, 1.0), "must be finite"),
        (get_dilution_step, (1.0, np.inf, 1.0), "must be finite"),
        (get_dilution_step, (1.0, 1.0, -1.0), "must be nonnegative"),
        (get_dilution_step, (1.0, 1.0, np.nan), "must be finite"),
        (get_dilution_step, (1.0, 1.0, np.inf), "must be finite"),
    ],
)
def test_dilution_helpers_reject_invalid_numeric_domains(
    function, arguments, message
):
    """Each numeric domain rejects zero where needed and nonfinite values."""
    with pytest.raises(ValueError, match=message):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (get_volume_dilution_coefficient, (np.array([1.0, np.nan]), 1.0)),
        (get_volume_dilution_coefficient, (1.0, np.array([1.0, -1.0]))),
        (get_dilution_rate, (np.array([1.0, np.inf]), 1.0)),
        (get_dilution_rate, (1.0, np.array([1.0, -1.0]))),
        (get_dilution_step, (np.array([1.0, -1.0]), 1.0, 1.0)),
        (get_dilution_step, (1.0, np.array([1.0, np.nan]), 1.0)),
        (get_dilution_step, (1.0, 1.0, np.array([1.0, np.inf]))),
    ],
)
def test_dilution_helpers_validate_every_array_element(function, arguments):
    """Invalid elements in otherwise compatible arrays fail validation."""
    with pytest.raises(ValueError):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments", "none_name"),
    [
        (get_volume_dilution_coefficient, (None, 1.0), "volume"),
        (get_volume_dilution_coefficient, (1.0, None), "input_flow_rate"),
        (get_dilution_rate, (None, 1.0), "coefficient"),
        (get_dilution_rate, (1.0, None), "concentration"),
        (get_dilution_step, (None, 1.0, 1.0), "coefficient"),
        (get_dilution_step, (1.0, None, 1.0), "concentration"),
        (get_dilution_step, (1.0, 1.0, None), "time_step"),
    ],
)
def test_dilution_helpers_reject_none(function, arguments, none_name):
    """Explicit None guards provide deterministic argument-specific errors."""
    with pytest.raises(
        TypeError, match=f"Argument '{none_name}' must not be None"
    ):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (get_volume_dilution_coefficient, ("one", 1.0)),
        (get_dilution_rate, (1.0, object())),
        (get_dilution_step, (1.0, "one", 1.0)),
    ],
)
def test_dilution_helpers_reject_unsupported_values(function, arguments):
    """Strings and object-valued inputs fail with TypeError."""
    with pytest.raises(TypeError):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (get_volume_dilution_coefficient, (np.ones((2, 2)), np.ones(3))),
        (get_dilution_rate, (np.ones((2, 2)), np.ones(3))),
        (get_dilution_step, (np.ones((2, 2)), np.ones(3), 1.0)),
    ],
)
def test_dilution_helpers_preflight_incompatible_shapes(function, arguments):
    """Incompatible operand shapes fail before arithmetic returns a result."""
    with pytest.raises(ValueError):
        function(*arguments)


@pytest.mark.parametrize(
    ("function", "arguments", "expected"),
    [
        (
            get_volume_dilution_coefficient,
            (np.array([2.0, 3.0]), 0.0),
            [0.0, 0.0],
        ),
        (get_dilution_rate, (0.0, np.array([2.0, 3.0])), [0.0, 0.0]),
        (get_dilution_step, (1.0, np.array([0.0, 0.0]), 2.0), [0.0, 0.0]),
        (
            get_dilution_step,
            (np.array([1.0, 2.0]), [2.0, 3.0], 0.0),
            [2.0, 3.0],
        ),
    ],
)
def test_dilution_no_ops_are_exact(function, arguments, expected):
    """Zero flow, coefficient, concentration, or duration preserve exact no-ops."""
    result = function(*arguments)
    npt.assert_array_equal(result, np.asarray(expected, dtype=np.float64))


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (
            get_volume_dilution_coefficient,
            (np.array([2.0, 4.0]), np.array([1.0, 2.0])),
        ),
        (get_dilution_rate, (np.array([0.5, 1.0]), np.array([2.0, 4.0]))),
        (get_dilution_step, (np.array([0.5, 1.0]), np.array([2.0, 4.0]), 1.0)),
    ],
)
def test_dilution_helpers_do_not_mutate_successful_inputs(function, arguments):
    """Successful vectorized operations leave caller-owned arrays unchanged."""
    snapshots = [
        value.copy() for value in arguments if isinstance(value, np.ndarray)
    ]
    function(*arguments)
    for value, snapshot in zip(
        (value for value in arguments if isinstance(value, np.ndarray)),
        snapshots,
        strict=True,
    ):
        npt.assert_array_equal(value, snapshot)


def test_dilution_helpers_do_not_mutate_failed_inputs():
    """Validation and broadcast failures leave their input arrays unchanged."""
    invalid = np.array([1.0, np.nan])
    incompatible = np.ones((2, 2))
    valid = np.ones(3)
    snapshots = (invalid.copy(), incompatible.copy(), valid.copy())

    with pytest.raises(ValueError):
        get_dilution_rate(1.0, invalid)
    with pytest.raises(ValueError):
        get_dilution_step(incompatible, valid, 1.0)

    npt.assert_array_equal(invalid, snapshots[0])
    npt.assert_array_equal(incompatible, snapshots[1])
    npt.assert_array_equal(valid, snapshots[2])


def test_dilution_package_surface_remains_limited_to_existing_helpers():
    """P1 keeps the finite-step helper concrete-module-only."""
    assert (
        dynamics.get_volume_dilution_coefficient
        is get_volume_dilution_coefficient
    )
    assert dynamics.get_dilution_rate is get_dilution_rate
    assert not hasattr(dynamics, "get_dilution_step")
    assert not hasattr(dynamics, "dilute_aerosol")


def _make_aerosol(n_species=1, n_boxes=1) -> Aerosol:
    """Build explicit resolved counts and independent full-gas lanes."""
    return Aerosol(
        particles=ParticleData(
            masses=np.tile(
                np.array([1e-18, 2e-18])[None, :, None], (n_boxes, 1, n_species)
            ),
            concentration=np.tile([[4.0, 8.0]], (n_boxes, 1)),
            charge=np.tile([[1.0, -1.0]], (n_boxes, 1)),
            density=np.full(n_species, 1000.0),
            volume=np.full(n_boxes, 2.0),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            name=["partitioning", "gas_a", "gas_b"],
            molar_mass=np.array([0.1, 0.02, 0.03]),
            concentration=np.tile([[3.0, 5.0, 7.0]], (n_boxes, 1)),
            partitioning=np.array([True, False, False]),
        ),
        environment=EnvironmentData(
            temperature=np.full(n_boxes, 298.15),
            pressure=np.full(n_boxes, 101325.0),
            saturation_ratio=np.ones((n_boxes, 3)),
        ),
    )


def _concentrations(aerosol):
    """Copy normalized particle concentration and the two gas categories."""
    return (
        (
            aerosol.particles.concentration[0] / aerosol.particles.volume[0]
        ).copy(),
        aerosol.gas.concentration[0, :1].copy(),
        aerosol.gas.concentration[0, 1:].copy(),
    )


def _storage(aerosol):
    """Return the authoritative concentration arrays."""
    return (aerosol.particles.concentration, aerosol.gas.concentration)


def _assert_unchanged(aerosol, snapshots):
    """Require exact unchanged primary concentration storage."""
    for actual, expected in zip(_storage(aerosol), snapshots, strict=True):
        npt.assert_array_equal(actual, expected)


def test_dilute_aerosol_updates_physical_particle_and_gas_concentrations():
    """All gas lanes and physical particle density decay with held identities."""
    aerosol = _make_aerosol()
    containers = (aerosol.particles, aerosol.gas, aerosol.environment)
    arrays = _storage(aerosol)
    sources = _concentrations(aerosol)
    assert dilute_aerosol(aerosol, 0.25, 4.0) is aerosol
    assert all(
        a is b
        for a, b in zip(
            containers,
            (aerosol.particles, aerosol.gas, aerosol.environment),
            strict=True,
        )
    )
    assert all(a is b for a, b in zip(arrays, _storage(aerosol), strict=True))
    for result, source in zip(_concentrations(aerosol), sources, strict=True):
        npt.assert_allclose(result, source * np.exp(-0.25 * 4.0))
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)


def test_dilute_aerosol_supports_speciated_particle_representation():
    """Dilution updates a supported speciated representation in physical units."""
    aerosol = _make_aerosol(n_species=2)
    source = (
        aerosol.particles.concentration[0] / aerosol.particles.volume[0]
    ).copy()

    dilute_aerosol(aerosol, 0.5, 2.0)

    npt.assert_allclose(
        (aerosol.particles.concentration[0] / aerosol.particles.volume[0]),
        source / np.e,
    )


def test_dilute_aerosol_preserves_container_metadata_and_particle_state():
    """Only concentrations change; every other native field is preserved."""
    aerosol = _make_aerosol(n_species=2)
    particle, gas, environment = (
        aerosol.particles,
        aerosol.gas,
        aerosol.environment,
    )
    fields = [
        particle.masses,
        particle.density,
        particle.charge,
        particle.volume,
        gas.molar_mass,
        gas.partitioning,
        environment.temperature,
        environment.pressure,
        environment.saturation_ratio,
    ]
    snapshots = [field.copy() for field in fields]
    density = particle.effective_density.copy()
    names = list(gas.name)
    dilute_aerosol(aerosol, 0.25, 4.0)
    for field, before in zip(fields, snapshots, strict=True):
        npt.assert_array_equal(field, before)
    npt.assert_array_equal(particle.effective_density, density)
    assert list(gas.name) == names
    assert particle.distribution_type == "particle_resolved"


@pytest.mark.parametrize("coefficient, time_step", [(0.0, 1.0), (1.0, 0.0)])
def test_dilute_aerosol_zero_coefficient_or_duration_is_exact_no_op(
    coefficient, time_step
):
    """Zero coefficient or duration retains all concentration values exactly."""
    aerosol = _make_aerosol()
    sources = (
        (
            aerosol.particles.concentration[0] / aerosol.particles.volume[0]
        ).copy(),
        np.asarray(aerosol.gas.concentration[0, :1]).copy(),
        aerosol.gas.concentration[0, 1:].copy(),
    )

    assert dilute_aerosol(aerosol, coefficient, time_step) is aerosol

    npt.assert_array_equal(
        (aerosol.particles.concentration[0] / aerosol.particles.volume[0]),
        sources[0],
    )
    npt.assert_array_equal(aerosol.gas.concentration[0, :1], sources[1])
    npt.assert_array_equal(aerosol.gas.concentration[0, 1:], sources[2])


@pytest.mark.parametrize("coefficient, time_step", [(0.0, 1.0), (1.0, 0.0)])
def test_dilute_aerosol_no_op_does_not_call_concentration_setters(
    monkeypatch, coefficient, time_step
):
    """A valid native no-op performs no copy into concentration storage."""
    aerosol = _make_aerosol()
    calls = []
    monkeypatch.setattr(
        dilution_module.np, "copyto", lambda *args: calls.append(args)
    )
    assert dilute_aerosol(aerosol, coefficient, time_step) is aerosol
    assert calls == []


def test_dilute_aerosol_zero_source_is_an_exact_no_op():
    """A canonical free population and zero gas remain exactly zero."""
    aerosol = _make_aerosol()
    aerosol.particles.concentration[:] = 0
    aerosol.particles.masses[:] = 0
    aerosol.particles.charge[:] = 0
    aerosol.gas.concentration[:] = 0
    snapshots = [x.copy() for x in _storage(aerosol)]
    dilute_aerosol(aerosol, 1.0, 2.0)
    _assert_unchanged(aerosol, snapshots)


@pytest.mark.parametrize(
    ("coefficient", "time_step", "exception", "message"),
    [
        (None, 1.0, TypeError, "coefficient.*numeric"),
        (1.0, None, TypeError, "time_step.*numeric"),
        ("invalid", 1.0, TypeError, "coefficient.*numeric"),
        (1.0, object(), TypeError, "time_step.*numeric"),
        (-1.0, 1.0, ValueError, "coefficient.*nonnegative"),
        (1.0, -1.0, ValueError, "time_step.*nonnegative"),
        (np.nan, 1.0, ValueError, "coefficient.*finite"),
        (1.0, np.nan, ValueError, "time_step.*finite"),
        (np.inf, 1.0, ValueError, "coefficient.*finite"),
        (1.0, np.inf, ValueError, "time_step.*finite"),
        (np.array([1.0]), 1.0, ValueError, "coefficient.*scalar"),
        (1.0, [1.0], ValueError, "time_step.*scalar"),
        (True, 1.0, TypeError, "coefficient.*boolean"),
        (False, 1.0, TypeError, "coefficient.*boolean"),
        (np.bool_(True), 1.0, TypeError, "coefficient.*boolean"),
        (np.bool_(False), 1.0, TypeError, "coefficient.*boolean"),
        (1.0, True, TypeError, "time_step.*boolean"),
        (1.0, False, TypeError, "time_step.*boolean"),
        (1.0, np.bool_(True), TypeError, "time_step.*boolean"),
        (1.0, np.bool_(False), TypeError, "time_step.*boolean"),
    ],
)
def test_dilute_aerosol_rejects_invalid_scalars_without_mutation(
    coefficient, time_step, exception, message
):
    """Scalar validation precedes every container access or assignment."""
    aerosol = _make_aerosol()
    snapshots = (
        (
            aerosol.particles.concentration[0] / aerosol.particles.volume[0]
        ).copy(),
        np.asarray(aerosol.gas.concentration[0, :1]).copy(),
        np.asarray(aerosol.gas.concentration[0, 1:]).copy(),
    )

    with pytest.raises(exception, match=message):
        dilute_aerosol(aerosol, coefficient, time_step)

    npt.assert_array_equal(
        (aerosol.particles.concentration[0] / aerosol.particles.volume[0]),
        snapshots[0],
    )
    npt.assert_array_equal(
        aerosol.gas.concentration[0, :1],
        snapshots[1],
    )
    npt.assert_array_equal(aerosol.gas.concentration[0, 1:], snapshots[2])


@pytest.mark.parametrize(
    "coefficient", [complex(0.5, 0.25), np.complex128(0.5 + 0.25j)]
)
def test_dilute_aerosol_rejects_complex_coefficient_warning_cleanly(
    coefficient,
):
    """Complex coefficients fail before conversion or concentration mutation."""
    aerosol = _make_aerosol()
    snapshots = (
        aerosol.particles.concentration.copy(),
        aerosol.gas.concentration[:, :1].copy(),
        aerosol.gas.concentration[:, 1:].copy(),
    )

    with pytest.raises(TypeError, match="coefficient.*real-valued"):
        dilute_aerosol(aerosol, coefficient, 1.0)

    npt.assert_array_equal(aerosol.particles.concentration, snapshots[0])
    npt.assert_array_equal(aerosol.gas.concentration[:, :1], snapshots[1])
    npt.assert_array_equal(aerosol.gas.concentration[:, 1:], snapshots[2])


def test_dilute_aerosol_rejects_multi_box_particles_before_mutation():
    """A structurally valid multi-box aggregate is rejected before writes."""
    aerosol = _make_aerosol(n_boxes=2)
    snapshots = [x.copy() for x in _storage(aerosol)]
    with pytest.raises(ValueError, match="exactly one box"):
        dilute_aerosol(aerosol, 0.5, 1.0)
    _assert_unchanged(aerosol, snapshots)


def test_dilute_aerosol_accepts_zero_dimensional_scalars_and_underflows():
    """Zero-dimensional numeric scalars are accepted and finite decay underflows."""
    aerosol = _make_aerosol()

    result = dilute_aerosol(
        aerosol,
        np.array(np.finfo(np.float64).max),
        np.array(np.finfo(np.float64).max),
    )

    assert result is aerosol
    npt.assert_array_equal(
        (aerosol.particles.concentration[0] / aerosol.particles.volume[0]),
        np.zeros(2),
    )
    assert aerosol.gas.concentration[0, :1] == 0.0
    npt.assert_array_equal(aerosol.gas.concentration[0, 1:], np.zeros(2))


def test_dilute_aerosol_preserves_subnormal_particle_storage():
    """Storage decay avoids erasing a representable count through volume scaling."""
    aerosol = _make_aerosol()
    aerosol.particles.concentration[0, 0] = np.nextafter(0.0, 1.0)
    aerosol.particles.volume[0] = 1e308

    dilute_aerosol(aerosol, 0.5, 1.0)

    assert aerosol.particles.concentration[0, 0] > 0.0


def test_dilute_aerosol_zero_dimensional_scalars_apply_normal_decay():
    """Zero-dimensional numeric inputs retain their documented scalar behavior."""
    aerosol = _make_aerosol()
    decay = np.exp(-np.float64(0.5) * np.float64(2.0))

    assert dilute_aerosol(aerosol, np.array(0.5), np.array(2.0)) is aerosol

    npt.assert_allclose(
        (aerosol.particles.concentration[0] / aerosol.particles.volume[0]),
        np.array([2.0, 4.0]) * decay,
    )
    npt.assert_allclose(aerosol.gas.concentration[0, :1], 3.0 * decay)
    npt.assert_allclose(
        aerosol.gas.concentration[0, 1:],
        np.array([5.0, 7.0]) * decay,
    )


@pytest.mark.parametrize(
    "invalid_group", ["particle", "partitioning", "gas_only"]
)
def test_dilute_aerosol_invalid_source_preflight_is_atomic(invalid_group):
    """A nonfinite concentration in any lane rejects without mutation."""
    aerosol = _make_aerosol()
    target = aerosol.particles if invalid_group == "particle" else aerosol.gas
    lane = 1 if invalid_group == "gas_only" else 0
    target.concentration[0, lane] = np.nan
    snapshots = [x.copy() for x in _storage(aerosol)]
    with pytest.raises(ValueError):
        dilute_aerosol(aerosol, 0.5, 1.0)
    _assert_unchanged(aerosol, snapshots)


@pytest.mark.parametrize(
    "invalid_group", ["particle", "partitioning", "gas_only"]
)
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, "invalid"])
@pytest.mark.parametrize("coefficient, time_step", [(0.5, 0.0), (0.0, 1.0)])
def test_dilute_aerosol_rejects_invalid_sources_before_writes(
    invalid_group, value, coefficient, time_step, monkeypatch
):
    """Even no-work calls validate every category before a writer is invoked."""
    aerosol = _make_aerosol()
    target = aerosol.particles if invalid_group == "particle" else aerosol.gas
    lane = 1 if invalid_group == "gas_only" else 0
    if isinstance(value, str):
        target.concentration = target.concentration.astype(object)
    target.concentration[0, lane] = value
    snapshots = [x.copy() for x in _storage(aerosol)]
    calls = []
    monkeypatch.setattr(
        dilution_module.np, "copyto", lambda *args: calls.append(args)
    )
    message = (
        (
            "particle data arrays must have float64 schema"
            if isinstance(value, str)
            else "particle data must have finite physical values"
        )
        if invalid_group == "particle"
        else "concentration"
    )
    exception = (
        TypeError
        if isinstance(value, str) and invalid_group != "particle"
        else ValueError
    )
    with pytest.raises(exception, match=message):
        dilute_aerosol(aerosol, coefficient, time_step)
    _assert_unchanged(aerosol, snapshots)
    assert calls == []


@pytest.mark.parametrize("call_index, candidate", [(0, np.ones(2)), (1, 1.0)])
def test_dilute_aerosol_rejects_candidate_shape_before_writes(
    monkeypatch, call_index, candidate
):
    """Both native update candidates must retain their original array shape."""
    aerosol = _make_aerosol()
    snapshots = [x.copy() for x in _storage(aerosol)]
    original_step = dilution_module.get_dilution_step
    calls = 0

    def malformed_step(*args):
        nonlocal calls
        result = candidate if calls == call_index else original_step(*args)
        calls += 1
        return result

    monkeypatch.setattr(dilution_module, "get_dilution_step", malformed_step)
    with pytest.raises(ValueError, match="concentration candidate"):
        dilute_aerosol(aerosol, 0.5, 1.0)
    _assert_unchanged(aerosol, snapshots)


@pytest.mark.parametrize(
    "volume",
    [
        np.array([np.nan]),
        np.array([0.0]),
        np.array([-1.0]),
        np.ones((1, 1)),
        np.array(["invalid"], dtype=object),
    ],
)
def test_dilute_aerosol_invalid_volume_preflight_is_atomic(volume):
    """Invalid physical volume or volume storage rejects before writes."""
    aerosol = _make_aerosol()
    aerosol.particles.volume = volume
    snapshots = [x.copy() for x in _storage(aerosol)]
    with pytest.raises(
        ValueError, match="volume|particle data.*(schema|physical)"
    ):
        dilute_aerosol(aerosol, 0.5, 1.0)
    _assert_unchanged(aerosol, snapshots)
    npt.assert_array_equal(aerosol.particles.volume, volume)


@pytest.mark.parametrize(
    "storage",
    [np.array([1.0]), np.array([["invalid", "invalid"]], dtype=object)],
)
def test_dilute_aerosol_rejects_invalid_particle_storage_before_writes(storage):
    """Native concentration shape and dtype failures preserve all storage."""
    aerosol = _make_aerosol()
    aerosol.particles.concentration = storage
    snapshots = [x.copy() for x in _storage(aerosol)]
    with pytest.raises(
        ValueError,
        match="concentration|particle data arrays must have float64 schema",
    ):
        dilute_aerosol(aerosol, 0.5, 1.0)
    _assert_unchanged(aerosol, snapshots)


@pytest.mark.parametrize(
    "storage", [np.array([-1.0, 2.0]), np.array([np.nan, 2.0])]
)
def test_dilute_aerosol_invalid_backing_storage_has_no_setter_writes(
    monkeypatch, storage
):
    """Malformed backing arrays reject before the native writer is called."""
    aerosol = _make_aerosol()
    aerosol.particles.concentration[0] = storage
    snapshots = [x.copy() for x in _storage(aerosol)]
    calls = []
    monkeypatch.setattr(
        dilution_module.np, "copyto", lambda *args: calls.append(args)
    )
    with pytest.raises(
        ValueError, match="particle data must have finite physical values"
    ):
        dilute_aerosol(aerosol, 0.5, 1.0)
    assert calls == []
    _assert_unchanged(aerosol, snapshots)


@pytest.mark.parametrize("fail_index", [0, 1])
@pytest.mark.parametrize("partial_write", [False, True])
def test_dilute_aerosol_restores_failed_native_writes(
    monkeypatch, fail_index, partial_write
):
    """Rollback restores all domains, including a partially written target."""
    aerosol = _make_aerosol()
    arrays = _storage(aerosol)
    snapshots = [x.copy() for x in arrays]
    original_copy = np.copyto
    calls = 0

    def copy_then_fail(target, candidate):
        nonlocal calls
        index = calls
        calls += 1
        if index == fail_index:
            if partial_write:
                target.flat[0] = candidate.flat[0]
            raise RuntimeError("native concentration write failure")
        original_copy(target, candidate)

    monkeypatch.setattr(dilution_module.np, "copyto", copy_then_fail)
    with pytest.raises(
        RuntimeError, match="native concentration write failure"
    ):
        dilute_aerosol(aerosol, 0.5, 1.0)
    _assert_unchanged(aerosol, snapshots)
    assert calls == fail_index + 1
    assert all(a is b for a, b in zip(arrays, _storage(aerosol), strict=True))


@pytest.mark.parametrize("operation", ["primitive", "rate", "step", "execute"])
def test_dilution_rejects_non_native_aggregate_before_access(operation):
    """Legacy-shaped objects cannot enter the supported native boundary."""
    from types import SimpleNamespace

    from particula.dynamics import Dilution, DilutionStrategy

    native = _make_aerosol()
    old_shape = SimpleNamespace(particles=native.particles, atmosphere=object())
    snapshots = [x.copy() for x in _storage(native)]
    strategy = DilutionStrategy(0.5)
    with pytest.raises(TypeError, match="dilution requires a native Aerosol"):
        if operation == "primitive":
            dilute_aerosol(old_shape, 0.5, 1.0)
        elif operation == "rate":
            strategy.rate(old_shape)
        elif operation == "step":
            strategy.step(old_shape, 1.0)
        else:
            Dilution(strategy).execute(old_shape, 1.0, 2)
    _assert_unchanged(native, snapshots)
