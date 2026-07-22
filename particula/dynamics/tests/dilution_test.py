"""Numerical-contract tests for chamber dilution helpers."""

import numpy as np
import numpy.testing as npt
import particula.dynamics as dynamics
import pytest
from particula.dynamics.dilution import (
    get_dilution_rate,
    get_dilution_step,
    get_volume_dilution_coefficient,
)


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
