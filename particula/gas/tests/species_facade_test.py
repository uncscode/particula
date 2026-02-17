"""Tests for GasSpecies facade over GasData.

Validates deprecation log messages, delegation, and internal data updates.
"""

import logging
from typing import cast

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.gas_data import GasData
from particula.gas.species import GasSpecies, _DEPRECATION_MESSAGE
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    VaporPressureStrategy,
)


def _make_single_box_data(
    name: str,
    molar_mass: float,
    concentration: float,
    partitioning: bool = True,
) -> GasData:
    """Create single-box GasData for a single species."""
    return GasData(
        name=[name],
        molar_mass=np.array([molar_mass], dtype=np.float64),
        concentration=np.array([[concentration]], dtype=np.float64),
        partitioning=np.array([partitioning], dtype=np.bool_),
    )


@pytest.fixture()
def _enable_particula_log_capture(caplog):
    """Allow caplog to capture the 'particula' logger (propagate=False)."""
    particula_logger = logging.getLogger("particula")
    old_propagate = particula_logger.propagate
    particula_logger.propagate = True
    caplog.set_level(logging.INFO, logger="particula")
    yield caplog
    particula_logger.propagate = old_propagate


def test_init_deprecation_log(_enable_particula_log_capture) -> None:
    """GasSpecies construction logs a deprecation info message."""
    caplog = _enable_particula_log_capture
    strategy = ConstantVaporPressureStrategy(2330.0)
    GasSpecies(
        name="Water",
        molar_mass=0.018,
        vapor_pressure_strategy=strategy,
        partitioning=True,
        concentration=1e-6,
    )
    assert _DEPRECATION_MESSAGE in caplog.text


def test_data_property_returns_gas_data() -> None:
    """Data property returns the wrapped GasData instance."""
    data = _make_single_box_data("Water", 0.018, 1e-6)
    strategy = ConstantVaporPressureStrategy(2330.0)
    species = GasSpecies.from_data(data, strategy)

    assert isinstance(species.data, GasData)


def test_from_data_no_deprecation_log(
    _enable_particula_log_capture,
) -> None:
    """from_data() does not emit the deprecation log message."""
    caplog = _enable_particula_log_capture
    data = _make_single_box_data("Water", 0.018, 1e-6)
    strategy = ConstantVaporPressureStrategy(2330.0)
    caplog.clear()
    _ = GasSpecies.from_data(data, strategy)
    assert _DEPRECATION_MESSAGE not in caplog.text


def test_facade_delegation_get_name() -> None:
    """Facade delegates name access to GasData."""
    data = _make_single_box_data("Water", 0.018, 1e-6)
    strategy = ConstantVaporPressureStrategy(2330.0)
    species = GasSpecies.from_data(data, strategy)

    assert species.get_name() == "Water"


def test_facade_delegation_get_molar_mass() -> None:
    """Facade delegates molar mass access to GasData."""
    data = _make_single_box_data("Water", 0.018, 1e-6)
    strategy = ConstantVaporPressureStrategy(2330.0)
    species = GasSpecies.from_data(data, strategy)

    assert species.get_molar_mass() == pytest.approx(0.018)


def test_facade_delegation_get_concentration() -> None:
    """Facade delegates concentration access to GasData."""
    data = _make_single_box_data("Water", 0.018, 1e-6)
    strategy = ConstantVaporPressureStrategy(2330.0)
    species = GasSpecies.from_data(data, strategy)

    assert species.get_concentration() == pytest.approx(1e-6)


def test_add_concentration_updates_internal_data() -> None:
    """add_concentration updates the internal GasData."""
    data = _make_single_box_data("Water", 0.018, 1e-6)
    strategy = ConstantVaporPressureStrategy(2330.0)
    species = GasSpecies.from_data(data, strategy)

    species.add_concentration(2e-6)

    assert species.data.concentration.shape == (1, 1)
    assert species.data.concentration[0, 0] == pytest.approx(3e-6)


def test_set_concentration_updates_internal_data() -> None:
    """set_concentration updates the internal GasData."""
    data = _make_single_box_data("Water", 0.018, 1e-6)
    strategy = ConstantVaporPressureStrategy(2330.0)
    species = GasSpecies.from_data(data, strategy)

    species.set_concentration(np.array([2e-6, 4e-6]))

    assert species.data.concentration.shape == (2, 1)
    npt.assert_allclose(
        species.data.concentration[:, 0], np.array([2e-6, 4e-6])
    )
    npt.assert_allclose(species.get_concentration(), np.array([2e-6, 4e-6]))


def test_concentration_property_setter_updates_internal_data() -> None:
    """Concentration property setter updates the internal GasData."""
    data = _make_single_box_data("Water", 0.018, 1e-6)
    strategy = ConstantVaporPressureStrategy(2330.0)
    species = GasSpecies.from_data(data, strategy)

    species.concentration = np.array([5e-6, 6e-6])

    assert species.data.concentration.shape == (2, 1)
    npt.assert_allclose(
        species.data.concentration[:, 0], np.array([5e-6, 6e-6])
    )
    npt.assert_allclose(species.get_concentration(), np.array([5e-6, 6e-6]))


def test_append_updates_internal_data() -> None:
    """Append updates GasData and extends strategies."""
    data1 = _make_single_box_data("Water", 0.018, 1e-6)
    data2 = _make_single_box_data("Ammonia", 0.017, 2e-6)
    strategy1 = ConstantVaporPressureStrategy(2330.0)
    strategy2 = ConstantVaporPressureStrategy(1000.0)

    species1 = GasSpecies.from_data(data1, strategy1)
    species2 = GasSpecies.from_data(data2, strategy2)

    species1.append(species2)

    assert species1.data.n_species == 2
    npt.assert_array_equal(
        np.asarray(species1.data.name, dtype=str),
        np.array(["Water", "Ammonia"]),
    )
    npt.assert_allclose(
        species1.data.concentration,
        np.array([[1e-6, 2e-6]]),
    )
    assert isinstance(species1.pure_vapor_pressure_strategy, list)
    assert len(species1.pure_vapor_pressure_strategy) == 2


def test_append_mismatched_box_counts_raises() -> None:
    """Append raises when both species have different box counts."""
    data1 = GasData(
        name=["Water"],
        molar_mass=np.array([0.018]),
        concentration=np.array([[1e-6], [2e-6]]),
        partitioning=np.array([True]),
    )
    data2 = GasData(
        name=["Ammonia"],
        molar_mass=np.array([0.017]),
        concentration=np.array([[3e-6], [4e-6], [5e-6]]),
        partitioning=np.array([True]),
    )
    strategy = ConstantVaporPressureStrategy(2330.0)

    species1 = GasSpecies.from_data(data1, strategy)
    species2 = GasSpecies.from_data(data2, strategy)

    with pytest.raises(ValueError, match="box dimensions"):
        species1.append(species2)


def test_multi_species_facade() -> None:
    """Facade works with multiple species in a single box."""
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
    species = GasSpecies.from_data(
        data, cast(list[VaporPressureStrategy], strategies)
    )

    npt.assert_array_equal(species.get_name(), np.array(["Water", "Ammonia"]))
    npt.assert_allclose(species.get_molar_mass(), np.array([0.018, 0.017]))
    npt.assert_allclose(species.get_concentration(), np.array([1e-6, 2e-6]))


def test_from_data_mixed_partitioning_raises() -> None:
    """from_data raises for mixed partitioning values."""
    data = GasData(
        name=["Water", "Ammonia"],
        molar_mass=np.array([0.018, 0.017]),
        concentration=np.array([[1e-6, 2e-6]]),
        partitioning=np.array([True, False]),
    )
    strategies: list[VaporPressureStrategy] = [
        ConstantVaporPressureStrategy(2330.0),
        ConstantVaporPressureStrategy(1000.0),
    ]

    with pytest.raises(ValueError, match="mixed partitioning"):
        GasSpecies.from_data(
            data, cast(list[VaporPressureStrategy], strategies)
        )
