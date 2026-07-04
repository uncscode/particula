"""Tests for the EnvironmentData dataclass."""

import importlib

import numpy as np
import particula.gas as gas_package
import pytest
from particula.gas.environment_data import EnvironmentData


def _make_environment_data(
    temperature: object,
    pressure: object,
    saturation_ratio: object,
) -> EnvironmentData:
    """Construct EnvironmentData with broad array-like test inputs."""
    return EnvironmentData(
        temperature=temperature,  # type: ignore[arg-type]
        pressure=pressure,  # type: ignore[arg-type]
        saturation_ratio=saturation_ratio,  # type: ignore[arg-type]
    )


def test_environment_data_valid_single_box_coerces_float64() -> None:
    """Single-box construction preserves shapes and coerces to float64."""
    environment = _make_environment_data(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[0.8, 1.2]],
    )

    assert environment.temperature.shape == (1,)
    assert environment.pressure.shape == (1,)
    assert environment.saturation_ratio.shape == (1, 2)
    assert environment.temperature.dtype == np.float64
    assert environment.pressure.dtype == np.float64
    assert environment.saturation_ratio.dtype == np.float64


def test_environment_data_valid_multi_box_preserves_expected_shapes() -> None:
    """Multi-box construction accepts matching leading dimensions."""
    environment = _make_environment_data(
        temperature=np.array([298.15, 300.0]),
        pressure=np.array([101325.0, 90000.0]),
        saturation_ratio=np.array([[0.8, 1.0], [1.2, 3.0]]),
    )

    assert environment.temperature.shape == (2,)
    assert environment.pressure.shape == (2,)
    assert environment.saturation_ratio.shape == (2, 2)


def test_environment_data_coerces_list_and_tuple_inputs_to_float64() -> None:
    """Lists and tuples are accepted for all array-backed fields."""
    environment = _make_environment_data(
        temperature=(298.15, 299.15),
        pressure=[101325.0, 100000.0],
        saturation_ratio=([0.9, 1.1], (0.5, 2.0)),
    )

    assert environment.temperature.dtype == np.float64
    assert environment.pressure.dtype == np.float64
    assert environment.saturation_ratio.dtype == np.float64


def test_environment_data_coerce_float64_array_returns_float64_array() -> None:
    """Private coercion helper converts array-like input to float64."""
    result = EnvironmentData._coerce_float64_array(
        [298.15, 299.15],
        field_name="temperature",
    )

    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.array([298.15, 299.15]))


def test_environment_data_coerce_float64_array_invalid_input_raises_value_error() -> (
    None
):
    """Private coercion helper rejects non-coercible array-like input."""
    with pytest.raises(
        ValueError,
        match="temperature must be array-like and coercible to float64",
    ):
        EnvironmentData._coerce_float64_array(
            ["not-a-number"],
            field_name="temperature",
        )


def test_environment_data_validate_dimensionality_accepts_valid_arrays() -> (
    None
):
    """Private dimensionality validator accepts valid array shapes."""
    environment = _make_environment_data(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[1.0]],
    )

    environment._validate_dimensionality()


def test_environment_data_validate_shapes_accepts_matching_box_counts() -> None:
    """Private shape validator accepts matching box counts."""
    environment = _make_environment_data(
        temperature=[298.15, 300.0],
        pressure=[101325.0, 100000.0],
        saturation_ratio=[[1.0], [0.9]],
    )

    environment._validate_shapes()


def test_environment_data_validate_finite_values_accepts_finite_arrays() -> (
    None
):
    """Private finite-value validator accepts finite numeric arrays."""
    environment = _make_environment_data(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[0.8, 1.2]],
    )

    environment._validate_finite_values()


def test_environment_data_validate_physical_bounds_accepts_valid_values() -> (
    None
):
    """Private bounds validator accepts positive and nonnegative values."""
    environment = _make_environment_data(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[0.0, 1.2]],
    )

    environment._validate_physical_bounds()


@pytest.mark.parametrize(
    "temperature",
    [298.15, None, [[298.15]]],
)
def test_environment_data_temperature_invalid_dimensionality_raises_value_error(
    temperature: object,
) -> None:
    """Temperature must be a 1D array."""
    with pytest.raises(ValueError, match="temperature must be 1D"):
        _make_environment_data(
            temperature=temperature,
            pressure=[101325.0],
            saturation_ratio=[[1.0]],
        )


@pytest.mark.parametrize(
    "pressure",
    [101325.0, None, [[101325.0]]],
)
def test_environment_data_pressure_invalid_dimensionality_raises_value_error(
    pressure: object,
) -> None:
    """Pressure must be a 1D array."""
    with pytest.raises(ValueError, match="pressure must be 1D"):
        _make_environment_data(
            temperature=[298.15],
            pressure=pressure,
            saturation_ratio=[[1.0]],
        )


@pytest.mark.parametrize(
    "saturation_ratio",
    [1.0, None, [1.0, 0.9], [[[1.0]]]],
)
def test_environment_data_saturation_ratio_invalid_dimensionality_raises_value_error(
    saturation_ratio: object,
) -> None:
    """Saturation ratio must be a 2D array."""
    with pytest.raises(ValueError, match="saturation_ratio must be 2D"):
        _make_environment_data(
            temperature=[298.15],
            pressure=[101325.0],
            saturation_ratio=saturation_ratio,
        )


def test_environment_data_temperature_pressure_length_mismatch_raises_value_error() -> (
    None
):
    """Pressure box count must match temperature box count."""
    with pytest.raises(
        ValueError,
        match="pressure shape must match temperature n_boxes",
    ):
        _make_environment_data(
            temperature=[298.15, 300.0],
            pressure=[101325.0],
            saturation_ratio=[[1.0], [0.9]],
        )


def test_environment_data_saturation_ratio_box_count_mismatch_raises_value_error() -> (
    None
):
    """Saturation ratio leading dimension must match temperature boxes."""
    with pytest.raises(
        ValueError,
        match="saturation_ratio leading dimension must match temperature",
    ):
        _make_environment_data(
            temperature=[298.15, 300.0],
            pressure=[101325.0, 90000.0],
            saturation_ratio=[[1.0, 0.9]],
        )


@pytest.mark.parametrize("nonfinite_value", [np.nan, np.inf, -np.inf])
def test_environment_data_nonfinite_temperature_raises_value_error(
    nonfinite_value: float,
) -> None:
    """Temperature must contain only finite values."""
    with pytest.raises(
        ValueError,
        match="temperature must contain only finite values",
    ):
        _make_environment_data(
            temperature=[nonfinite_value],
            pressure=[101325.0],
            saturation_ratio=[[1.0]],
        )


@pytest.mark.parametrize("nonfinite_value", [np.nan, np.inf, -np.inf])
def test_environment_data_nonfinite_pressure_raises_value_error(
    nonfinite_value: float,
) -> None:
    """Pressure must contain only finite values."""
    with pytest.raises(
        ValueError,
        match="pressure must contain only finite values",
    ):
        _make_environment_data(
            temperature=[298.15],
            pressure=[nonfinite_value],
            saturation_ratio=[[1.0]],
        )


@pytest.mark.parametrize("nonfinite_value", [np.nan, np.inf, -np.inf])
def test_environment_data_nonfinite_saturation_ratio_raises_value_error(
    nonfinite_value: float,
) -> None:
    """Saturation ratio must contain only finite values."""
    with pytest.raises(
        ValueError,
        match="saturation_ratio must contain only finite values",
    ):
        _make_environment_data(
            temperature=[298.15],
            pressure=[101325.0],
            saturation_ratio=[[nonfinite_value]],
        )


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_environment_data_nonpositive_temperature_raises_value_error(
    temperature: float,
) -> None:
    """Temperature must be strictly positive."""
    with pytest.raises(
        ValueError, match="temperature must be strictly positive"
    ):
        _make_environment_data(
            temperature=[temperature],
            pressure=[101325.0],
            saturation_ratio=[[1.0]],
        )


@pytest.mark.parametrize("pressure", [0.0, -1.0])
def test_environment_data_nonpositive_pressure_raises_value_error(
    pressure: float,
) -> None:
    """Pressure must be strictly positive."""
    with pytest.raises(ValueError, match="pressure must be strictly positive"):
        _make_environment_data(
            temperature=[298.15],
            pressure=[pressure],
            saturation_ratio=[[1.0]],
        )


@pytest.mark.parametrize("saturation_ratio", [-1.0, -1e-6])
def test_environment_data_negative_saturation_ratio_raises_value_error(
    saturation_ratio: float,
) -> None:
    """Saturation ratio must be nonnegative."""
    with pytest.raises(
        ValueError, match="saturation_ratio must be nonnegative"
    ):
        _make_environment_data(
            temperature=[298.15],
            pressure=[101325.0],
            saturation_ratio=[[saturation_ratio]],
        )


def test_environment_data_supersaturation_values_above_one_are_valid() -> None:
    """Finite supersaturation values above one remain valid."""
    environment = _make_environment_data(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[1.2, 3.0]],
    )

    np.testing.assert_allclose(
        environment.saturation_ratio,
        np.array([[1.2, 3.0]], dtype=np.float64),
    )


def test_environment_data_direct_module_import_requires_no_package_export_change() -> (
    None
):
    """Direct module import works without adding a package-level export."""
    module = importlib.import_module("particula.gas.environment_data")

    assert module.EnvironmentData is EnvironmentData
    assert not hasattr(gas_package, "EnvironmentData")
