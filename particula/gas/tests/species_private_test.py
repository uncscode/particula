"""Tests for GasSpecies private helpers."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.gas_data import GasData
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    VaporPressureStrategy,
)


def _make_facade_single() -> GasSpecies:
    """Create a single-species facade without deprecation warning."""
    data = GasData(
        name=["Water"],
        molar_mass=np.array([0.018]),
        concentration=np.array([[1e-6]]),
        partitioning=np.array([True]),
    )
    strategy = ConstantVaporPressureStrategy(2330.0)
    return GasSpecies.from_data(data, strategy)


def _make_facade_multi() -> GasSpecies:
    """Create a multi-species facade without deprecation warning."""
    data = GasData(
        name=["Water", "Ammonia"],
        molar_mass=np.array([0.018, 0.017]),
        concentration=np.array([[1e-6, 2e-6]]),
        partitioning=np.array([True, True]),
    )
    strategies: list[VaporPressureStrategy] = [
        ConstantVaporPressureStrategy(2330.0),
        ConstantVaporPressureStrategy(1000.0),
    ]
    return GasSpecies.from_data(data, strategies)


def test_infer_species_count_prefers_max_dimension() -> None:
    """_infer_species_count returns the maximum input species count."""
    species = _make_facade_single()
    count = species._infer_species_count(
        name="Water",
        molar_mass=np.array([0.018, 0.017]),
        concentration=np.array([1e-6, 2e-6]),
    )

    assert count == 2


def test_normalize_names_str_and_array_modes() -> None:
    """_normalize_names returns list and mode for inputs."""
    species = _make_facade_single()

    names, mode = species._normalize_names("Water", n_species=2)
    assert names == ["Water", "Water"]
    assert mode == "array"

    names, mode = species._normalize_names(
        np.array(["Water", "Ammonia"]), n_species=2
    )
    assert names == ["Water", "Ammonia"]
    assert mode == "array"


def test_normalize_molar_mass_scalar_and_mismatch_error() -> None:
    """_normalize_molar_mass expands scalars and validates sizes."""
    species = _make_facade_single()

    values, mode = species._normalize_molar_mass(0.018, n_species=2)
    npt.assert_allclose(values, np.array([0.018, 0.018]))
    assert mode == "scalar"

    with pytest.raises(ValueError, match="molar_mass length"):
        species._normalize_molar_mass(np.array([0.018, 0.017]), n_species=3)


def test_normalize_concentration_scalar_and_array_modes() -> None:
    """_normalize_concentration handles scalar and array inputs."""
    species = _make_facade_single()

    values, mode = species._normalize_concentration(1e-6, n_species=1)
    assert values.shape == (1, 1)
    assert mode == "scalar"

    values, mode = species._normalize_concentration(
        np.array([1e-6, 2e-6]), n_species=1
    )
    assert values.shape == (2, 1)
    assert mode == "array"


def test_normalize_concentration_multi_box_raises() -> None:
    """_normalize_concentration raises for multi-box multi-species input."""
    species = _make_facade_multi()
    with pytest.raises(ValueError, match="single box"):
        species._normalize_concentration(
            np.array([[1e-6, 2e-6], [3e-6, 4e-6]]), n_species=2
        )


def test_normalize_partitioning_returns_boolean_array() -> None:
    """_normalize_partitioning returns a boolean array mask."""
    species = _make_facade_single()

    values = species._normalize_partitioning(False, n_species=3)

    npt.assert_array_equal(values, np.array([False, False, False]))


def test_set_concentration_rebuilds_data() -> None:
    """_set_concentration rebuilds GasData and updates modes."""
    species = _make_facade_single()

    species._set_concentration(np.array([2e-6, 4e-6]))

    assert species.data.concentration.shape == (2, 1)
    assert species._single_species_concentration_mode == "array"
    npt.assert_allclose(
        species.data.concentration[:, 0], np.array([2e-6, 4e-6])
    )


def test_species_arrays_for_strategy_single_and_multi() -> None:
    """_species_arrays_for_strategy returns correct arrays."""
    single = _make_facade_single()
    conc, molar = single._species_arrays_for_strategy()
    npt.assert_allclose(conc, np.array([1e-6]))
    npt.assert_allclose(molar, np.array([0.018]))

    multi = _make_facade_multi()
    conc, molar = multi._species_arrays_for_strategy()
    npt.assert_allclose(conc, np.array([1e-6, 2e-6]))
    npt.assert_allclose(molar, np.array([0.018, 0.017]))


def test_check_if_negative_concentration_clamps_and_warns() -> None:
    """_check_if_negative_concentration clamps to zero and warns."""
    species = _make_facade_single()

    with pytest.warns(UserWarning, match="Negative concentration"):
        values = species._check_if_negative_concentration(np.array([-1.0, 2.0]))

    npt.assert_allclose(values, np.array([0.0, 2.0]))


def test_check_non_positive_value_raises() -> None:
    """_check_non_positive_value raises for non-positive values."""
    species = _make_facade_single()

    with pytest.raises(ValueError, match="Non-positive"):
        species._check_non_positive_value(0.0, name="molar_mass")
