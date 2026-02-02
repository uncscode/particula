"""Tests for the ParticleData dataclass.

Tests cover instantiation with valid shapes, edge cases (single box,
single particle, single species), and validation error cases for
shape mismatches.
"""

import numpy as np
import pytest
from particula.particles.particle_data import ParticleData


class TestParticleDataInstantiation:
    """Tests for valid ParticleData instantiation."""

    def test_valid_single_box(self) -> None:
        """Test valid instantiation with single box (n_boxes=1)."""
        data = ParticleData(
            masses=np.random.rand(1, 5, 3) * 1e-18,
            concentration=np.ones((1, 5)),
            charge=np.zeros((1, 5)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6]),
        )
        assert data.masses.shape == (1, 5, 3)
        assert data.concentration.shape == (1, 5)
        assert data.charge.shape == (1, 5)
        assert data.density.shape == (3,)
        assert data.volume.shape == (1,)

    def test_valid_multi_box(self) -> None:
        """Test valid instantiation with multiple boxes (n_boxes=10)."""
        data = ParticleData(
            masses=np.zeros((10, 100, 5)),
            concentration=np.ones((10, 100)),
            charge=np.zeros((10, 100)),
            density=np.array([1000.0, 1100.0, 1200.0, 1300.0, 1400.0]),
            volume=np.ones(10) * 1e-6,
        )
        assert data.masses.shape == (10, 100, 5)
        assert data.concentration.shape == (10, 100)
        assert data.charge.shape == (10, 100)
        assert data.density.shape == (5,)
        assert data.volume.shape == (10,)

    def test_valid_single_particle(self) -> None:
        """Test edge case: single particle (n_particles=1)."""
        data = ParticleData(
            masses=np.ones((2, 1, 3)) * 1e-18,
            concentration=np.ones((2, 1)),
            charge=np.zeros((2, 1)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6, 2e-6]),
        )
        assert data.masses.shape == (2, 1, 3)
        assert data.concentration.shape == (2, 1)

    def test_valid_single_species(self) -> None:
        """Test edge case: single species (n_species=1)."""
        data = ParticleData(
            masses=np.ones((2, 5, 1)) * 1e-18,
            concentration=np.ones((2, 5)),
            charge=np.zeros((2, 5)),
            density=np.array([1000.0]),
            volume=np.array([1e-6, 2e-6]),
        )
        assert data.masses.shape == (2, 5, 1)
        assert data.density.shape == (1,)

    def test_attributes_accessible(self) -> None:
        """Verify all attributes accessible after instantiation."""
        masses = np.random.rand(1, 10, 2) * 1e-18
        concentration = np.ones((1, 10))
        charge = np.zeros((1, 10))
        density = np.array([1000.0, 1500.0])
        volume = np.array([1e-6])

        data = ParticleData(
            masses=masses,
            concentration=concentration,
            charge=charge,
            density=density,
            volume=volume,
        )

        # Verify all attributes are accessible and have correct values
        np.testing.assert_array_equal(data.masses, masses)
        np.testing.assert_array_equal(data.concentration, concentration)
        np.testing.assert_array_equal(data.charge, charge)
        np.testing.assert_array_equal(data.density, density)
        np.testing.assert_array_equal(data.volume, volume)


class TestParticleDataValidation:
    """Tests for ParticleData validation errors."""

    def test_masses_not_3d(self) -> None:
        """Validation error when masses is 2D instead of 3D."""
        with pytest.raises(
            ValueError,
            match="masses must be 3D",
        ):
            ParticleData(
                masses=np.ones((1, 5)),  # 2D instead of 3D
                concentration=np.ones((1, 5)),
                charge=np.zeros((1, 5)),
                density=np.array([1000.0]),
                volume=np.array([1e-6]),
            )

    def test_concentration_shape_wrong_n_boxes(self) -> None:
        """Validation error when concentration has wrong n_boxes."""
        with pytest.raises(
            ValueError,
            match="concentration.*shape",
        ):
            ParticleData(
                masses=np.ones((2, 5, 3)),  # n_boxes=2
                concentration=np.ones((3, 5)),  # wrong n_boxes=3
                charge=np.ones((2, 5)),
                density=np.ones(3),
                volume=np.ones(2),
            )

    def test_concentration_shape_wrong_n_particles(self) -> None:
        """Validation error when concentration has wrong n_particles."""
        with pytest.raises(
            ValueError,
            match="concentration.*shape",
        ):
            ParticleData(
                masses=np.ones((2, 5, 3)),  # n_particles=5
                concentration=np.ones((2, 10)),  # wrong n_particles=10
                charge=np.ones((2, 5)),
                density=np.ones(3),
                volume=np.ones(2),
            )

    def test_charge_shape_wrong_n_boxes(self) -> None:
        """Validation error when charge has wrong n_boxes."""
        with pytest.raises(
            ValueError,
            match="charge.*shape",
        ):
            ParticleData(
                masses=np.ones((2, 5, 3)),  # n_boxes=2
                concentration=np.ones((2, 5)),
                charge=np.ones((3, 5)),  # wrong n_boxes=3
                density=np.ones(3),
                volume=np.ones(2),
            )

    def test_charge_shape_wrong_n_particles(self) -> None:
        """Validation error when charge has wrong n_particles."""
        with pytest.raises(
            ValueError,
            match="charge.*shape",
        ):
            ParticleData(
                masses=np.ones((2, 5, 3)),  # n_particles=5
                concentration=np.ones((2, 5)),
                charge=np.ones((2, 10)),  # wrong n_particles=10
                density=np.ones(3),
                volume=np.ones(2),
            )

    def test_volume_shape_mismatch(self) -> None:
        """Validation error for volume with wrong n_boxes."""
        with pytest.raises(
            ValueError,
            match="volume.*shape",
        ):
            ParticleData(
                masses=np.ones((2, 5, 3)),  # n_boxes=2
                concentration=np.ones((2, 5)),
                charge=np.ones((2, 5)),
                density=np.ones(3),
                volume=np.ones(3),  # wrong n_boxes=3
            )

    def test_density_not_1d(self) -> None:
        """Validation error when density is not 1D."""
        with pytest.raises(
            ValueError,
            match="density must be 1D",
        ):
            ParticleData(
                masses=np.ones((2, 5, 3)),
                concentration=np.ones((2, 5)),
                charge=np.ones((2, 5)),
                density=np.ones((1, 3)),  # 2D instead of 1D
                volume=np.ones(2),
            )

    def test_n_species_mismatch(self) -> None:
        """Validation error when n_species mismatches masses vs density."""
        with pytest.raises(
            ValueError,
            match="n_species",
        ):
            ParticleData(
                masses=np.ones((2, 5, 3)),  # n_species=3
                concentration=np.ones((2, 5)),
                charge=np.ones((2, 5)),
                density=np.ones(2),  # wrong n_species=2
                volume=np.ones(2),
            )
