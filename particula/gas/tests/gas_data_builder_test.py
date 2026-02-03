"""Tests for GasDataBuilder covering conversions, shapes, and validation."""

# ruff: noqa: D102

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.gas_data import GasData
from particula.gas.gas_data_builder import GasDataBuilder

# pint is optional; skip tests that require it if not installed
pint = pytest.importorskip("pint")


class TestGasDataBuilderBasics:
    """Basic builds and required fields."""

    def test_build_valid_single_box(self) -> None:
        """Builder creates valid GasData with all fields set."""
        data = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass(np.array([0.018, 0.017]), units="kg/mol")
            .set_concentration(np.array([1e15, 1e12]), units="1/m^3")
            .set_partitioning(np.array([True, False]))
            .build()
        )
        assert isinstance(data, GasData)
        assert data.n_boxes == 1
        assert data.n_species == 2
        assert data.concentration.shape == (1, 2)
        assert data.molar_mass.shape == (2,)
        assert data.partitioning.shape == (2,)
        npt.assert_allclose(data.molar_mass, np.array([0.018, 0.017]))
        npt.assert_allclose(data.concentration, np.array([[1e15, 1e12]]))

    def test_fluent_chaining(self) -> None:
        """All setters return self for chaining."""
        data = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass([0.018])
            .set_concentration([1e15])
            .set_partitioning([True])
            .build()
        )
        assert isinstance(data, GasData)
        assert data.n_species == 1

    def test_multi_box_build(self) -> None:
        """Builder with 2D concentration array creates multi-box GasData."""
        concentration_2d = np.array([[1e15, 1e12], [2e15, 2e12]])
        data = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass(np.array([0.018, 0.017]))
            .set_concentration(concentration_2d)
            .set_partitioning([True, True])
            .build()
        )
        assert data.n_boxes == 2
        assert data.concentration.shape == (2, 2)
        npt.assert_allclose(data.concentration, concentration_2d)


class TestGasDataBuilderUnits:
    """Unit conversion coverage."""

    @pytest.mark.parametrize(
        "value, units, expected",
        [
            (np.array([18.0]), "g/mol", 0.018),
            (np.array([0.018]), "kg/mol", 0.018),
        ],
    )
    def test_molar_mass_units(
        self, value: np.ndarray, units: str, expected: float
    ) -> None:
        """Molar mass conversion from g/mol to kg/mol."""
        data = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass(value, units=units)
            .set_concentration([1e15])
            .set_partitioning([True])
            .build()
        )
        npt.assert_allclose(data.molar_mass[0], expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "value, units, expected",
        [
            (np.array([1e9]), "1/cm^3", 1e15),
            (np.array([1e15]), "1/m^3", 1e15),
        ],
    )
    def test_concentration_units(
        self, value: np.ndarray, units: str, expected: float
    ) -> None:
        """Concentration conversion from 1/cm^3 to 1/m^3."""
        data = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass([0.018])
            .set_concentration(value, units=units)
            .set_partitioning([True])
            .build()
        )
        npt.assert_allclose(data.concentration[0, 0], expected, rtol=1e-6)


class TestGasDataBuilderBatch:
    """Batch dimension handling."""

    def test_1d_concentration_gets_batch_dim(self) -> None:
        """1D concentration array gets batch dimension added."""
        data = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_concentration([1e15, 1e12])
            .set_partitioning([True, True])
            .build()
        )
        assert data.concentration.shape == (1, 2)

    def test_2d_concentration_unchanged(self) -> None:
        """2D concentration array used as-is."""
        concentration_2d = np.array([[1e15, 1e12], [2e15, 2e12]])
        data = (
            GasDataBuilder()
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_concentration(concentration_2d)
            .set_partitioning([True, True])
            .build()
        )
        assert data.concentration.shape == (2, 2)

    def test_n_boxes_broadcasts_1d(self) -> None:
        """set_n_boxes broadcasts 1D concentration to n_boxes."""
        data = (
            GasDataBuilder()
            .set_n_boxes(100)
            .set_names(["Water", "Ammonia"])
            .set_molar_mass([0.018, 0.017])
            .set_concentration([1e15, 1e12])
            .set_partitioning([True, True])
            .build()
        )
        assert data.concentration.shape == (100, 2)
        # All rows should be the same
        npt.assert_allclose(data.concentration[0], data.concentration[99])


class TestGasDataBuilderValidation:
    """Validation and error propagation."""

    def test_missing_names_raises(self) -> None:
        """ValueError when names not set."""
        builder = (
            GasDataBuilder()
            .set_molar_mass([0.018])
            .set_concentration([1e15])
            .set_partitioning([True])
        )
        with pytest.raises(ValueError, match="names is required"):
            builder.build()

    def test_missing_molar_mass_raises(self) -> None:
        """ValueError when molar_mass not set."""
        builder = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_concentration([1e15])
            .set_partitioning([True])
        )
        with pytest.raises(ValueError, match="molar_mass is required"):
            builder.build()

    def test_missing_concentration_raises(self) -> None:
        """ValueError when concentration not set."""
        builder = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass([0.018])
            .set_partitioning([True])
        )
        with pytest.raises(ValueError, match="concentration is required"):
            builder.build()

    def test_missing_partitioning_raises(self) -> None:
        """ValueError when partitioning not set."""
        builder = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass([0.018])
            .set_concentration([1e15])
        )
        with pytest.raises(ValueError, match="partitioning is required"):
            builder.build()

    def test_negative_molar_mass_raises(self) -> None:
        """ValueError when molar_mass has values <= 0."""
        builder = GasDataBuilder()
        with pytest.raises(ValueError, match="positive"):
            builder.set_molar_mass([-0.018])

    def test_zero_molar_mass_raises(self) -> None:
        """ValueError when molar_mass is exactly 0."""
        builder = GasDataBuilder()
        with pytest.raises(ValueError, match="positive"):
            builder.set_molar_mass([0.0])

    def test_negative_concentration_raises(self) -> None:
        """ValueError when concentration is negative."""
        builder = GasDataBuilder()
        with pytest.raises(ValueError, match="non-negative"):
            builder.set_concentration([-1e15])

    def test_molar_mass_not_1d_raises(self) -> None:
        """ValueError when molar_mass is 2D."""
        builder = GasDataBuilder()
        with pytest.raises(ValueError, match="must be 1D"):
            builder.set_molar_mass(np.array([[0.018, 0.017]]))

    def test_concentration_3d_raises(self) -> None:
        """ValueError when concentration is 3D."""
        builder = GasDataBuilder()
        with pytest.raises(ValueError, match="1D or 2D"):
            builder.set_concentration(np.ones((2, 3, 4)))

    def test_partitioning_not_1d_raises(self) -> None:
        """ValueError when partitioning is 2D."""
        builder = GasDataBuilder()
        with pytest.raises(ValueError, match="must be 1D"):
            builder.set_partitioning(np.array([[True, False]]))

    def test_invalid_unit_raises(self) -> None:
        """UndefinedUnitError for unknown units."""
        builder = GasDataBuilder()
        with pytest.raises(pint.errors.UndefinedUnitError):
            builder.set_molar_mass([18.0], units="invalid_unit")


class TestGasDataBuilderDtype:
    """dtype enforcement."""

    def test_numeric_outputs_are_float64(self) -> None:
        """molar_mass and concentration should be float64."""
        data = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass([0.018])
            .set_concentration([1e15])
            .set_partitioning([True])
            .build()
        )
        assert data.molar_mass.dtype == np.float64
        assert data.concentration.dtype == np.float64

    def test_partitioning_is_bool(self) -> None:
        """Partitioning should be bool_."""
        data = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass([0.018])
            .set_concentration([1e15])
            .set_partitioning([True])
            .build()
        )
        assert data.partitioning.dtype == np.bool_

    def test_int_input_converted_to_float(self) -> None:
        """Integer arrays should be converted to float64."""
        data = (
            GasDataBuilder()
            .set_names(["H2O"])
            .set_molar_mass(np.array([1], dtype=np.int32), units="g/mol")
            .set_concentration(
                np.array([1000000000], dtype=np.int64), units="1/cm^3"
            )
            .set_partitioning([True])  # truthy value
            .build()
        )
        assert data.molar_mass.dtype == np.float64
        assert data.concentration.dtype == np.float64
        assert data.partitioning.dtype == np.bool_
