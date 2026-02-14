"""Tests for the GasData dataclass."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.gas_data import GasData, from_species, to_species
from particula.gas.species import GasSpecies
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)


class TestGasDataInstantiation:
    """Tests for valid GasData instantiation and accessors."""

    def test_valid_single_box(self) -> None:
        """Test valid instantiation with single box (n_boxes=1)."""
        gas = GasData(
            name=["Water", "Ammonia", "H2SO4"],
            molar_mass=np.array([0.018, 0.017, 0.098]),
            concentration=np.array([[1e-6, 5e-9, 2e-10]]),
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
            concentration=np.array([[1e-7], [2e-7]]),
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
        """Partitioning is coerced to boolean dtype."""
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
            concentration=np.array([[1e-6, 5e-9], [2e-6, 1e-8]]),
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
            concentration=np.array([[1e-6, 5e-9, 2e-10]]),
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
            concentration=np.array([[1e-6, 5e-9]]),
            partitioning=np.array([True, False]),
        )

        gas_copy = gas.copy()
        gas.name[0] = "Changed"

        assert gas_copy.name[0] == "Water"


class TestFromSpecies:
    """Tests for from_species() conversion utility."""

    def test_single_species_conversion(self) -> None:
        """Convert single-species GasSpecies to GasData."""
        vapor_pressure = ConstantVaporPressureStrategy(2330.0)
        species = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=vapor_pressure,
            partitioning=True,
            concentration=1e-6,  # kg/m^3
        )

        gas_data = from_species(species)

        assert gas_data.n_boxes == 1
        assert gas_data.n_species == 1
        assert gas_data.name == ["Water"]
        npt.assert_allclose(gas_data.molar_mass, np.array([0.018]))
        assert gas_data.concentration.shape == (1, 1)
        np.testing.assert_array_equal(gas_data.partitioning, np.array([True]))

    def test_multi_species_conversion(self) -> None:
        """Convert multi-species GasSpecies (built with +=) to GasData."""
        vp1 = ConstantVaporPressureStrategy(2330.0)
        vp2 = ConstantVaporPressureStrategy(1000.0)

        species1 = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=vp1,
            partitioning=True,
            concentration=1e-6,
        )
        species2 = GasSpecies(
            name="Ammonia",
            molar_mass=0.017,
            vapor_pressure_strategy=vp2,
            partitioning=True,
            concentration=2e-6,
        )
        species1 += species2

        gas_data = from_species(species1)

        assert gas_data.n_boxes == 1
        assert gas_data.n_species == 2
        assert gas_data.name == ["Water", "Ammonia"]
        npt.assert_allclose(gas_data.molar_mass, np.array([0.018, 0.017]))
        assert gas_data.concentration.shape == (1, 2)
        np.testing.assert_array_equal(
            gas_data.partitioning, np.array([True, True])
        )

    def test_n_boxes_replication(self) -> None:
        """from_species with n_boxes>1 replicates concentration correctly."""
        vapor_pressure = ConstantVaporPressureStrategy(2330.0)
        species = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=vapor_pressure,
            partitioning=True,
            concentration=1e-6,
        )

        gas_data = from_species(species, n_boxes=5)

        assert gas_data.n_boxes == 5
        assert gas_data.n_species == 1
        assert gas_data.concentration.shape == (5, 1)
        # All boxes should have the same concentration
        for i in range(5):
            npt.assert_allclose(
                gas_data.concentration[i, :], gas_data.concentration[0, :]
            )

    def test_concentration_passthrough(self) -> None:
        """Verify kg/m^3 concentrations are copied without conversion."""
        molar_mass = 0.018  # kg/mol (water)
        concentration_kg = 1e-6  # kg/m^3

        vapor_pressure = ConstantVaporPressureStrategy(2330.0)
        species = GasSpecies(
            name="Water",
            molar_mass=molar_mass,
            vapor_pressure_strategy=vapor_pressure,
            partitioning=True,
            concentration=concentration_kg,
        )

        gas_data = from_species(species)

        npt.assert_allclose(gas_data.concentration[0, 0], concentration_kg)


class TestToSpecies:
    """Tests for to_species() conversion utility."""

    def test_single_species_conversion(self) -> None:
        """Convert GasData to single-species GasSpecies."""
        gas_data = GasData(
            name=["Water"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e-6]]),  # kg/m^3
            partitioning=np.array([True]),
        )
        strategy = ConstantVaporPressureStrategy(2330.0)

        species = to_species(gas_data, [strategy])

        assert species.get_name() == "Water"
        npt.assert_allclose(species.get_molar_mass(), 0.018)
        assert species.get_partitioning() is True
        npt.assert_allclose(species.get_concentration(), 1e-6, rtol=1e-10)

    def test_multi_species_conversion(self) -> None:
        """Convert GasData to multi-species GasSpecies."""
        gas_data = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e-6, 2e-6]]),  # kg/m^3
            partitioning=np.array([True, True]),
        )
        strategies = [
            ConstantVaporPressureStrategy(2330.0),
            ConstantVaporPressureStrategy(1000.0),
        ]

        species = to_species(gas_data, strategies)

        assert len(species) == 2
        np.testing.assert_array_equal(
            species.get_name(), np.array(["Water", "Ammonia"])
        )
        npt.assert_allclose(species.get_molar_mass(), np.array([0.018, 0.017]))
        assert species.get_partitioning() is True

    def test_box_index_selection(self) -> None:
        """Verify box_index parameter selects correct concentration."""
        gas_data = GasData(
            name=["Water"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e-6], [2e-6], [3e-6]]),  # 3 boxes
            partitioning=np.array([True]),
        )
        strategy = ConstantVaporPressureStrategy(2330.0)

        species_box0 = to_species(gas_data, [strategy], box_index=0)
        species_box1 = to_species(gas_data, [strategy], box_index=1)
        species_box2 = to_species(gas_data, [strategy], box_index=2)

        npt.assert_allclose(species_box0.get_concentration(), 1e-6, rtol=1e-10)
        npt.assert_allclose(species_box1.get_concentration(), 2e-6, rtol=1e-10)
        npt.assert_allclose(species_box2.get_concentration(), 3e-6, rtol=1e-10)

    def test_strategy_length_mismatch_raises(self) -> None:
        """ValueError when strategies length doesn't match n_species."""
        gas_data = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e-6, 2e-6]]),
            partitioning=np.array([True, True]),
        )
        strategy = ConstantVaporPressureStrategy(2330.0)

        with pytest.raises(ValueError, match="doesn't match n_species"):
            to_species(gas_data, [strategy])  # Only 1 strategy for 2 species

    def test_box_index_out_of_range_raises(self) -> None:
        """IndexError when box_index >= n_boxes."""
        gas_data = GasData(
            name=["Water"],
            molar_mass=np.array([0.018]),
            concentration=np.array([[1e-6], [2e-6]]),  # 2 boxes
            partitioning=np.array([True]),
        )
        strategy = ConstantVaporPressureStrategy(2330.0)

        with pytest.raises(IndexError, match="out of range"):
            to_species(gas_data, [strategy], box_index=2)  # Only 0, 1 valid

    def test_mixed_partitioning_raises(self) -> None:
        """ValueError when GasData has mixed partitioning values."""
        gas_data = GasData(
            name=["Water", "Ammonia"],
            molar_mass=np.array([0.018, 0.017]),
            concentration=np.array([[1e-6, 2e-6]]),
            partitioning=np.array([True, False]),  # Mixed!
        )
        strategies = [
            ConstantVaporPressureStrategy(2330.0),
            ConstantVaporPressureStrategy(1000.0),
        ]

        with pytest.raises(ValueError, match="mixed partitioning"):
            to_species(gas_data, strategies)


class TestRoundTrip:
    """Tests for round-trip conversion GasSpecies -> GasData -> GasSpecies."""

    def test_round_trip_single_species(self) -> None:
        """Round-trip conversion preserves data within precision."""
        vapor_pressure = ConstantVaporPressureStrategy(2330.0)
        original = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=vapor_pressure,
            partitioning=True,
            concentration=1e-6,  # kg/m^3
        )

        # Convert to GasData and back
        gas_data = from_species(original)
        recovered = to_species(gas_data, [vapor_pressure])

        assert recovered.get_name() == original.get_name()
        npt.assert_allclose(
            recovered.get_molar_mass(), original.get_molar_mass(), rtol=1e-10
        )
        npt.assert_allclose(
            recovered.get_concentration(),
            original.get_concentration(),
            rtol=1e-10,
        )
        assert recovered.get_partitioning() == original.get_partitioning()

    def test_round_trip_multi_species(self) -> None:
        """Round-trip preserves data for multi-species."""
        vp1 = ConstantVaporPressureStrategy(2330.0)
        vp2 = ConstantVaporPressureStrategy(1000.0)

        species1 = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=vp1,
            partitioning=True,
            concentration=1e-6,
        )
        species2 = GasSpecies(
            name="Ammonia",
            molar_mass=0.017,
            vapor_pressure_strategy=vp2,
            partitioning=True,
            concentration=2e-6,
        )
        original = species1 + species2  # Creates new combined species

        # Convert to GasData and back
        gas_data = from_species(original)
        recovered = to_species(gas_data, [vp1, vp2])

        assert len(recovered) == len(original)
        np.testing.assert_array_equal(recovered.get_name(), original.get_name())
        npt.assert_allclose(
            recovered.get_molar_mass(), original.get_molar_mass(), rtol=1e-10
        )
        npt.assert_allclose(
            recovered.get_concentration(),
            original.get_concentration(),
            rtol=1e-10,
        )
        assert recovered.get_partitioning() == original.get_partitioning()

    def test_round_trip_with_different_partitioning(self) -> None:
        """Test round-trip with partitioning=False (uniform)."""
        vapor_pressure = ConstantVaporPressureStrategy(2330.0)
        original = GasSpecies(
            name="Water",
            molar_mass=0.018,
            vapor_pressure_strategy=vapor_pressure,
            partitioning=False,  # Different from True
            concentration=1e-6,
        )

        # Convert to GasData and back
        gas_data = from_species(original)
        recovered = to_species(gas_data, [vapor_pressure])

        assert recovered.get_partitioning() is False
        npt.assert_allclose(
            recovered.get_concentration(),
            original.get_concentration(),
            rtol=1e-10,
        )
