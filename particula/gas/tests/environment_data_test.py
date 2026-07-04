"""Tests for the EnvironmentData dataclass."""

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


def test_environment_data_n_boxes_property() -> None:
    """Box count is derived from the validated temperature axis."""
    single_box_environment = _make_environment_data(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[0.8, 1.2]],
    )
    multi_box_environment = _make_environment_data(
        temperature=[298.15, 300.0],
        pressure=[101325.0, 90000.0],
        saturation_ratio=[[0.8, 1.0], [1.2, 3.0]],
    )

    assert single_box_environment.n_boxes == 1
    assert multi_box_environment.n_boxes == 2


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


@pytest.mark.parametrize(
    "saturation_ratio",
    [np.empty((0, 0)), np.empty((0, 2))],
)
def test_environment_data_zero_box_inputs_raise_value_error(
    saturation_ratio: np.ndarray,
) -> None:
    """EnvironmentData requires at least one box."""
    with pytest.raises(ValueError, match="requires at least one box"):
        _make_environment_data(
            temperature=[],
            pressure=[],
            saturation_ratio=saturation_ratio,
        )


class _OverflowOnFloat:
    """Helper object that raises OverflowError during float coercion."""

    def __float__(self) -> float:
        raise OverflowError("float overflow during coercion")


@pytest.mark.parametrize(
    ("field_name", "temperature", "pressure", "saturation_ratio"),
    [
        (
            "temperature",
            [[1.0], [2.0, 3.0]],
            [101325.0],
            [[1.0]],
        ),
        (
            "pressure",
            [298.15],
            [_OverflowOnFloat()],
            [[1.0]],
        ),
        (
            "saturation_ratio",
            [298.15],
            [101325.0],
            [[object()]],
        ),
    ],
)
def test_environment_data_invalid_coercion_raises_value_error(
    field_name: str,
    temperature: object,
    pressure: object,
    saturation_ratio: object,
) -> None:
    """Malformed constructor inputs fail with normalized ValueError."""
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be array-like and coercible to float64",
    ):
        _make_environment_data(
            temperature=temperature,
            pressure=pressure,
            saturation_ratio=saturation_ratio,
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


def test_environment_data_valid_inputs_preserve_float64_values() -> None:
    """Valid constructor inputs remain accepted and coerced to float64."""
    environment = _make_environment_data(
        temperature=np.array([298.15, 300.15], dtype=np.float32),
        pressure=(101325, 95000),
        saturation_ratio=[[0, 1.0], [1.2, 0.5]],
    )

    assert environment.temperature.dtype == np.float64
    assert environment.pressure.dtype == np.float64
    assert environment.saturation_ratio.dtype == np.float64
    np.testing.assert_allclose(
        environment.temperature,
        np.array([298.15, 300.15], dtype=np.float64),
    )


def test_environment_data_copy_creates_independent_arrays() -> None:
    """copy() preserves values while duplicating array storage."""
    environment = _make_environment_data(
        temperature=[298.15, 300.15],
        pressure=[101325.0, 95000.0],
        saturation_ratio=[[0.9, 1.0], [1.2, 0.5]],
    )

    environment_copy = environment.copy()

    assert environment_copy is not environment
    np.testing.assert_allclose(
        environment_copy.temperature,
        environment.temperature,
    )
    np.testing.assert_allclose(
        environment_copy.pressure,
        environment.pressure,
    )
    np.testing.assert_allclose(
        environment_copy.saturation_ratio,
        environment.saturation_ratio,
    )
    assert not np.shares_memory(
        environment_copy.temperature,
        environment.temperature,
    )
    assert not np.shares_memory(
        environment_copy.pressure,
        environment.pressure,
    )
    assert not np.shares_memory(
        environment_copy.saturation_ratio,
        environment.saturation_ratio,
    )


def test_environment_data_copy_mutation_does_not_change_source() -> None:
    """Mutating copied arrays leaves the original instance unchanged."""
    environment = _make_environment_data(
        temperature=[298.15, 300.15],
        pressure=[101325.0, 95000.0],
        saturation_ratio=[[0.9, 1.0], [1.2, 0.5]],
    )

    environment_copy = environment.copy()
    environment_copy.temperature[0] = 310.0
    environment_copy.pressure[1] = 97000.0
    environment_copy.saturation_ratio[0, 1] = 0.75

    np.testing.assert_allclose(
        environment.temperature,
        np.array([298.15, 300.15], dtype=np.float64),
    )
    np.testing.assert_allclose(
        environment.pressure,
        np.array([101325.0, 95000.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        environment.saturation_ratio,
        np.array([[0.9, 1.0], [1.2, 0.5]], dtype=np.float64),
    )


def test_environment_data_comparison_uses_identity_semantics() -> None:
    """Distinct instances compare deterministically without NumPy ambiguity."""
    environment = _make_environment_data(
        temperature=[298.15, 300.15],
        pressure=[101325.0, 95000.0],
        saturation_ratio=[[0.9, 1.0], [1.2, 0.5]],
    )
    matching_environment = _make_environment_data(
        temperature=[298.15, 300.15],
        pressure=[101325.0, 95000.0],
        saturation_ratio=[[0.9, 1.0], [1.2, 0.5]],
    )

    assert environment == environment
    assert environment != matching_environment


def test_environment_data_is_exported_from_particula_gas() -> None:
    """Package and direct-module imports resolve to the same class."""
    assert gas_package.EnvironmentData is EnvironmentData
