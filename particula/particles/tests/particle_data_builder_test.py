"""Tests for ParticleDataBuilder covering conversions, shapes,
and validation.
"""

# ruff: noqa: D102

import numpy as np
import pytest
from particula.particles.particle_data import ParticleData
from particula.particles.particle_data_builder import ParticleDataBuilder

# pint is optional; skip tests that require it if not installed
pint = pytest.importorskip("pint")


class TestParticleDataBuilderBasics:
    """Basic builds and required fields."""

    def test_basic_build(self) -> None:
        masses = np.array([[1e-18, 2e-18]])
        data = (
            ParticleDataBuilder()
            .set_masses(masses, units="kg")
            .set_density(np.array([1000.0, 1200.0]), units="kg/m^3")
            .build()
        )
        assert isinstance(data, ParticleData)
        assert data.masses.shape == (1, 1, 2)
        np.testing.assert_allclose(data.masses[0, 0], masses[0])
        np.testing.assert_allclose(data.density, np.array([1000.0, 1200.0]))
        np.testing.assert_allclose(data.concentration, np.ones((1, 1)))
        np.testing.assert_allclose(data.charge, np.zeros((1, 1)))
        np.testing.assert_allclose(data.volume, np.ones(1))

    def test_missing_masses_requires_counts(self) -> None:
        builder = ParticleDataBuilder().set_density(
            np.array([1000.0]), units="kg/m^3"
        )
        with pytest.raises(ValueError, match="n_boxes"):
            builder.build()

    def test_missing_density_raises(self) -> None:
        builder = ParticleDataBuilder().set_masses(
            np.array([[1e-18]]), units="kg"
        )
        with pytest.raises(ValueError, match="density"):
            builder.build()


class TestParticleDataBuilderUnits:
    """Unit conversion coverage."""

    @pytest.mark.parametrize(
        "value, units, expected",
        [
            (np.array([[1.0]]), "kg", 1.0),
            (np.array([[1.0]]), "g", 1e-3),
            (np.array([[1.0]]), "ug", 1e-9),
            (np.array([[1.0]]), "ng", 1e-12),
        ],
    )
    def test_mass_units(
        self, value: np.ndarray, units: str, expected: float
    ) -> None:
        data = (
            ParticleDataBuilder()
            .set_masses(value, units=units)
            .set_density(np.array([1000.0]), units="kg/m^3")
            .build()
        )
        np.testing.assert_allclose(data.masses, expected)

    @pytest.mark.parametrize(
        "value, units, expected",
        [
            (np.array([1000.0]), "kg/m^3", 1000.0),
            (np.array([1.0]), "g/cm^3", 1000.0),
        ],
    )
    def test_density_units(
        self, value: np.ndarray, units: str, expected: float
    ) -> None:
        data = (
            ParticleDataBuilder()
            .set_masses(np.array([[1e-18]]), units="kg")
            .set_density(value, units=units)
            .build()
        )
        np.testing.assert_allclose(data.density, expected)

    @pytest.mark.parametrize(
        "value, units, expected",
        [
            (np.array([1.0]), "m^3", 1.0),
            (np.array([1.0]), "cm^3", 1e-6),
            (np.array([1.0]), "L", 1e-3),
        ],
    )
    def test_volume_units(
        self, value: np.ndarray, units: str, expected: float
    ) -> None:
        data = (
            ParticleDataBuilder()
            .set_masses(np.array([[1e-18]]), units="kg")
            .set_density(np.array([1000.0]), units="kg/m^3")
            .set_volume(value, units=units)
            .build()
        )
        np.testing.assert_allclose(data.volume, expected)

    @pytest.mark.parametrize(
        "value, units, expected",
        [
            (np.array([1.0]), "1/m^3", 1.0),
            (np.array([1.0]), "1/cm^3", 1e6),
        ],
    )
    def test_concentration_units(
        self, value: np.ndarray, units: str, expected: float
    ) -> None:
        data = (
            ParticleDataBuilder()
            .set_masses(np.array([[1e-18]]), units="kg")
            .set_density(np.array([1000.0]), units="kg/m^3")
            .set_concentration(value, units=units)
            .build()
        )
        np.testing.assert_allclose(data.concentration, expected)


class TestParticleDataBuilderBatch:
    """Batch dimension handling."""

    def test_masses_2d_auto_batch(self) -> None:
        masses = np.array([[1e-18, 2e-18]])  # (n_particles, n_species)
        data = (
            ParticleDataBuilder()
            .set_masses(masses, units="kg")
            .set_density(np.array([1000.0, 1200.0]), units="kg/m^3")
            .build()
        )
        assert data.masses.shape == (1, 1, 2)

    def test_concentration_1d_auto_batch(self) -> None:
        masses = np.zeros((1, 2, 1))
        data = (
            ParticleDataBuilder()
            .set_masses(masses, units="kg")
            .set_density(np.array([1000.0]), units="kg/m^3")
            .set_concentration(np.array([1.0, 2.0]), units="1/m^3")
            .build()
        )
        assert data.concentration.shape == (1, 2)

    def test_charge_1d_auto_batch(self) -> None:
        masses = np.zeros((1, 2, 1))
        data = (
            ParticleDataBuilder()
            .set_masses(masses, units="kg")
            .set_density(np.array([1000.0]), units="kg/m^3")
            .set_charge(np.array([0.0, 1.0]))
            .build()
        )
        assert data.charge.shape == (1, 2)

    def test_volume_scalar_broadcast(self) -> None:
        masses = np.zeros((2, 3, 1))
        data = (
            ParticleDataBuilder()
            .set_masses(masses, units="kg")
            .set_density(np.array([1000.0]), units="kg/m^3")
            .set_volume(2.0, units="m^3")
            .build()
        )
        np.testing.assert_allclose(data.volume, np.array([2.0, 2.0]))


class TestParticleDataBuilderZeroInit:
    """Zero-initialization path when masses are absent."""

    def test_zero_init_with_counts(self) -> None:
        data = (
            ParticleDataBuilder()
            .set_n_boxes(2)
            .set_n_particles(3)
            .set_n_species(2)
            .set_density(np.array([1000.0, 1200.0]), units="kg/m^3")
            .build()
        )
        assert data.masses.shape == (2, 3, 2)
        assert np.all(data.masses == 0.0)
        np.testing.assert_allclose(data.concentration, np.ones((2, 3)))
        np.testing.assert_allclose(data.charge, np.zeros((2, 3)))
        np.testing.assert_allclose(data.volume, np.ones(2))

    def test_zero_init_requires_species(self) -> None:
        builder = (
            ParticleDataBuilder()
            .set_n_boxes(1)
            .set_n_particles(2)
            .set_density(np.array([1000.0]), units="kg/m^3")
        )
        with pytest.raises(ValueError, match="n_species"):
            builder.build()

    def test_zero_init_requires_density(self) -> None:
        builder = (
            ParticleDataBuilder()
            .set_n_boxes(1)
            .set_n_particles(2)
            .set_n_species(1)
        )
        with pytest.raises(ValueError, match="density"):
            builder.build()


class TestParticleDataBuilderValidation:
    """Validation and error propagation."""

    def test_masses_shape_mismatch_with_counts(self) -> None:
        masses = np.ones((1, 2, 1))
        builder = (
            ParticleDataBuilder()
            .set_masses(masses, units="kg")
            .set_density(np.array([1000.0]), units="kg/m^3")
            .set_n_boxes(2)
        )
        with pytest.raises(ValueError, match="masses shape mismatch"):
            builder.build()

    def test_particle_data_shape_validation_propagates(self) -> None:
        masses = np.ones((1, 2, 1))
        builder = (
            ParticleDataBuilder()
            .set_masses(masses, units="kg")
            .set_density(np.array([1000.0]), units="kg/m^3")
        )
        # Force wrong concentration shape to trip ParticleData validation
        builder._concentration = np.ones((2, 3))  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="concentration shape"):
            builder.build()

    def test_invalid_unit_raises(self) -> None:
        builder = ParticleDataBuilder()
        with pytest.raises(pint.errors.UndefinedUnitError):
            builder.set_masses(np.array([[1.0]]), units="invalid")


class TestParticleDataBuilderDtype:
    """dtype enforcement."""

    def test_numeric_outputs_are_float64(self) -> None:
        data = (
            ParticleDataBuilder()
            .set_masses(np.array([[1]], dtype=np.int64), units="kg")
            .set_density(np.array([1000], dtype=np.int64), units="kg/m^3")
            .set_concentration(np.array([1], dtype=np.int32))
            .set_charge(np.array([0], dtype=np.int32))
            .set_volume(np.array([1], dtype=np.int32), units="m^3")
            .build()
        )
        assert data.masses.dtype == np.float64
        assert data.concentration.dtype == np.float64
        assert data.charge.dtype == np.float64
        assert data.density.dtype == np.float64
        assert data.volume.dtype == np.float64
