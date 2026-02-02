"""Tests for the GasData dataclass."""

import numpy as np
import numpy.testing as npt
import pytest

from particula.gas.gas_data import GasData


class TestGasDataInstantiation:
    """Tests for valid GasData instantiation and accessors."""

    def test_valid_single_box(self) -> None:
        """Test valid instantiation with single box (n_boxes=1)."""
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e15, 1e12, 1e10]]),
            partitioning=np.array([True, True, True]),
        )

        assert gas.n_boxes == 1
        assert gas.n_species == 3
        assert gas.concentration.shape == (1, 3)
        assert gas.molar_mass.shape == (3,)
        assert gas.partitioning.shape == (3,)

    def test_valid_multi_box(self) -> None:
        """Test valid instantiation with multiple boxes (n_boxes>1)."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.zeros((4, 2)),
            partitioning=np.array([True, False]),
        )

        assert gas.n_boxes == 4
        assert gas.n_species == 2
        assert gas.concentration.shape == (4, 2)
        assert gas.molar_mass.shape == (2,)
        assert gas.partitioning.shape == (2,)

    def test_valid_single_species(self) -> None:
        """Test edge case: single species (n_species=1)."""
        gas = GasData(
            name=["Water"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e12], [2e12]]),
            partitioning=np.array([True]),
        )

        assert gas.n_boxes == 2
        assert gas.n_species == 1
        assert gas.concentration.shape == (2, 1)
        assert gas.molar_mass.shape == (1,)
        assert gas.partitioning.shape == (1,)


class TestGasDataValidation:
    """Tests for GasData validation errors."""

    def test_molar_mass_shape_mismatch(self) -> None:
        """Validation error when molar_mass has wrong shape."""
        with pytest.raises(ValueError, match="molar_mass shape"):
            GasData(
                name=["Water", "Ammonia"],
                molar_mass=np.array([[0.018, 0.017]]),  # wrong shape
                concentration=np.zeros((1, 2)),
                partitioning=np.array([True, True]),
            )

    def test_partitioning_shape_mismatch(self) -> None:
        """Validation error when partitioning has wrong shape."""
        with pytest.raises(ValueError, match="partitioning shape"):
            GasData(
                name=["Water", "Ammonia"],
                molar_mass=np.array([0.018, 0.017]),
                concentration=np.zeros((1, 2)),
                partitioning=np.array([[True, False]]),  # wrong shape
            )

    def test_partitioning_casts_to_bool(self) -> None:
        """partitioning is coerced to boolean dtype."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.zeros((1, 2)),
            partitioning=np.array([1, 0]),  # convertible to bool
        )

        assert gas.partitioning.dtype == np.bool_
        np.testing.assert_array_equal(gas.partitioning, np.array([True, False]))

    def test_concentration_not_2d(self) -> None:
        """Validation error when concentration is not 2D."""
        with pytest.raises(ValueError, match="must be 2D"):
            GasData(
                name=["Water", "Ammonia"],
                molar_mass=np.array([0.018, 0.017]),
                concentration=np.array([1e15, 1e12]),  # 1D instead of 2D
                partitioning=np.array([True, True]),
            )

    def test_concentration_n_species_mismatch(self) -> None:
        """Validation error when concentration n_species doesn't match names."""
        with pytest.raises(ValueError, match="n_species dimension"):
            GasData(
                name=["Water", "Ammonia", "H2SO4"],
                molar_mass=np.array([0.018, 0.017, 0.098]),
                concentration=np.zeros((2, 2)),  # width 2 vs 3 names
                partitioning=np.array([True, True, True]),
            )

    def test_empty_name_raises(self) -> None:
        """Empty name list raises ValueError."""
        with pytest.raises(ValueError, match="at least one species"):
            GasData(
                name=[],
                molar_mass=np.array([], dtype=np.float64),
                concentration=np.zeros((1, 0)),
                partitioning=np.array([], dtype=bool),
            )


class TestGasDataProperties:
    """Tests for GasData properties."""

    def test_n_boxes_property(self) -> None:
        """n_boxes returns correct value."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.zeros((5, 2)),
            partitioning=np.array([True, True]),
        )

        assert gas.n_boxes == 5

    def test_n_species_property(self) -> None:
        """n_species returns correct value."""
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.zeros((3, 3)),
            partitioning=np.array([True, True, True]),
        )

        assert gas.n_species == 3


class TestGasDataCopy:
    """Tests for GasData copy method."""

    def test_copy_creates_independent_arrays(self) -> None:
        """copy() returns new arrays that do not share memory."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e15, 1e12], [2e15, 2e12]]),
            partitioning=np.array([True, False]),
        )

        gas_copy = gas.copy()

        assert not np.shares_memory(gas.molar_mass, gas_copy.molar_mass)
        assert not np.shares_memory(gas.concentration, gas_copy.concentration)
        assert not np.shares_memory(gas.partitioning, gas_copy.partitioning)

    def test_copy_preserves_values(self) -> None:
        """copy() preserves all values."""
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e15, 1e12, 1e10]]),
            partitioning=np.array([True, False, True]),
        )

        gas_copy = gas.copy()

        npt.assert_allclose(gas_copy.molar_mass, gas.molar_mass)
        npt.assert_allclose(gas_copy.concentration, gas.concentration)
        np.testing.assert_array_equal(gas_copy.partitioning, gas.partitioning)

    def test_copy_independent_name_list(self) -> None:
        """copy() produces independent name list."""
        gas = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e15, 1e12]]),
            partitioning=np.array([True, False]),
        )

        gas_copy = gas.copy()
        gas.name[0] = "Changed"

        assert gas_copy.name[0] == "Water"
