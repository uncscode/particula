"""Tests for the GasData dataclass."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.gas_data import GasData


@pytest.mark.parametrize("boxes", [1, 3, 5])
@pytest.mark.parametrize(
    "partitioning",
    [[True], [False], [True, True], [False, False], [True, False]],
)
@pytest.mark.parametrize("distinct_boxes", [False, True])
def test_native_storage_preserves_bridge_fields(
    boxes, partitioning, distinct_boxes
):
    """Ordered native storage/copy retains every full-gas lane without scaling."""
    species = len(partitioning)
    names = ["Water", "Ammonia"][:species]
    molar_mass = np.array([0.018, 0.017])[:species]
    concentration = np.tile(np.array([1e-6, 2e-6])[:species], (boxes, 1))
    if distinct_boxes:
        concentration *= np.arange(1, boxes + 1)[:, None]
    original = GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=np.array(partitioning),
    )
    copied = original.copy()
    for gas in (original, copied):
        assert gas.n_boxes == boxes
        assert gas.n_species == species
        assert gas.name == names
        assert gas.concentration.shape == (boxes, species)
        npt.assert_array_equal(gas.molar_mass, molar_mass)
        npt.assert_array_equal(gas.partitioning, partitioning)
        for box in range(boxes):
            npt.assert_allclose(
                gas.concentration[box], concentration[box], rtol=1e-10, atol=0
            )
    assert copied.name is not original.name
    for name in ("molar_mass", "concentration", "partitioning"):
        assert not np.shares_memory(
            getattr(original, name), getattr(copied, name)
        )
    copied.concentration[:] = 0
    npt.assert_array_equal(original.concentration, concentration)


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

    @pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
    @pytest.mark.parametrize("mask", [[True, False, True], [False] * 3])
    def test_full_order_and_extensive_mass(
        self, volume: float, mask: list[bool]
    ) -> None:
        """Copy retains every gas lane; gas inventory is C times V."""
        gas = GasData(
            name=["water", "inert", "organic"],
            molar_mass=np.array([0.018, 0.028, 0.12]),
            concentration=np.array([[1e-6, 2e-6, 3e-6]]),
            partitioning=np.array(mask),
        )
        copied = gas.copy()
        assert copied.name == ["water", "inert", "organic"]
        assert copied.name is not gas.name
        for field in ("molar_mass", "concentration", "partitioning"):
            npt.assert_array_equal(getattr(copied, field), getattr(gas, field))
            assert not np.shares_memory(
                getattr(copied, field), getattr(gas, field)
            )
        expected = {
            0.25: [0.25e-6, 0.5e-6, 0.75e-6],
            1.0: [1e-6, 2e-6, 3e-6],
            4.0: [4e-6, 8e-6, 12e-6],
        }[volume]
        npt.assert_allclose(
            gas.concentration[0] * volume, expected, rtol=1e-12, atol=1e-30
        )
        gas.name[1] = "changed"
        gas.concentration[0, 1] = 0.0
        assert copied.name[1] == "inert"
        assert copied.concentration[0, 1] == 2e-6

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
