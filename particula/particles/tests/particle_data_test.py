"""Tests for the ParticleData dataclass.

Tests cover instantiation with valid shapes, edge cases (single box,
single particle, single species), and validation error cases for
shape mismatches.
"""

import numpy as np
import pytest
from particula.particles.particle_data import ParticleData


class TestParticleDataInstantiation:
    """Tests for valid ParticleData instantiation and accessors."""

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

    def test_dimension_properties(self) -> None:
        """Test n_boxes, n_particles, n_species properties."""
        data = ParticleData(
            masses=np.zeros((5, 100, 3)),
            concentration=np.ones((5, 100)),
            charge=np.zeros((5, 100)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.ones(5),
        )
        assert data.n_boxes == 5
        assert data.n_particles == 100
        assert data.n_species == 3


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


class TestParticleDataProperties:
    """Tests for computed properties of ParticleData."""

    def test_radii_single_species(self) -> None:
        """Radii matches analytic 1 µm sphere for single species."""
        mass = (4.0 / 3.0) * np.pi * (1e-6) ** 3 * 1000.0
        data = ParticleData(
            masses=np.array([[[mass]]]),
            concentration=np.array([[1.0]]),
            charge=np.array([[0.0]]),
            density=np.array([1000.0]),
            volume=np.array([1e-6]),
        )
        np.testing.assert_allclose(data.radii, [[1e-6]], rtol=1e-12)
        assert data.radii.shape == (1, 1)

    def test_radii_multi_species(self) -> None:
        """Radii computed from aggregated volume across species."""
        masses = np.array([[[1e-18, 2e-18, 3e-18]]])
        density = np.array([1000.0, 1200.0, 800.0])
        data = ParticleData(
            masses=masses,
            concentration=np.array([[1.0]]),
            charge=np.array([[0.0]]),
            density=density,
            volume=np.array([1e-6]),
        )
        volumes_per_species = masses / density
        total_volume = np.sum(volumes_per_species, axis=-1)
        expected_radius = np.cbrt(3.0 * total_volume / (4.0 * np.pi))
        np.testing.assert_allclose(data.radii, expected_radius, rtol=1e-12)
        assert data.radii.shape == (1, 1)

    def test_radii_zero_mass_returns_zero(self) -> None:
        """Radii is zero when total mass is zero."""
        data = ParticleData(
            masses=np.zeros((2, 3, 2)),
            concentration=np.ones((2, 3)),
            charge=np.zeros((2, 3)),
            density=np.array([1000.0, 1200.0]),
            volume=np.ones(2),
        )
        np.testing.assert_array_equal(data.radii, np.zeros((2, 3)))
        assert data.radii.shape == (2, 3)

    def test_total_mass(self) -> None:
        """Total mass sums across species."""
        data = ParticleData(
            masses=np.array([[[1e-18, 2e-18, 3e-18], [4e-18, 5e-18, 6e-18]]]),
            concentration=np.ones((1, 2)),
            charge=np.zeros((1, 2)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6]),
        )
        np.testing.assert_allclose(data.total_mass, [[6e-18, 15e-18]])
        assert data.total_mass.shape == (1, 2)

    def test_effective_density_single_species(self) -> None:
        """Effective density equals species density for single species."""
        data = ParticleData(
            masses=np.array([[[1e-18]]]),
            concentration=np.ones((1, 1)),
            charge=np.zeros((1, 1)),
            density=np.array([1500.0]),
            volume=np.array([1e-6]),
        )
        np.testing.assert_allclose(data.effective_density, [[1500.0]])
        assert data.effective_density.shape == (1, 1)

    def test_effective_density_multi_species(self) -> None:
        """Effective density is mass-weighted average density across species."""
        masses = np.array([[[1e-18, 2e-18, 3e-18]]])
        density = np.array([1000.0, 1200.0, 800.0])
        data = ParticleData(
            masses=masses,
            concentration=np.array([[1.0]]),
            charge=np.array([[0.0]]),
            density=density,
            volume=np.array([1e-6]),
        )
        # Match ParticleRepresentation.get_effective_density formula:
        # sum(mass_i * density_i) / sum(mass_i)
        mass_weighted_density = np.sum(masses * density, axis=-1)
        total_mass = np.sum(masses, axis=-1)
        expected_density = mass_weighted_density / total_mass
        np.testing.assert_allclose(data.effective_density, expected_density)
        assert data.effective_density.shape == (1, 1)

    def test_effective_density_zero_mass(self) -> None:
        """Effective density returns zero when total mass is zero."""
        data = ParticleData(
            masses=np.zeros((1, 2, 2)),
            concentration=np.ones((1, 2)),
            charge=np.zeros((1, 2)),
            density=np.array([1000.0, 1200.0]),
            volume=np.array([1e-6]),
        )
        np.testing.assert_array_equal(data.effective_density, [[0.0, 0.0]])
        assert data.effective_density.shape == (1, 2)

    def test_mass_fractions_sum_to_one(self) -> None:
        """Mass fractions sum to one per particle when mass present."""
        data = ParticleData(
            masses=np.array([[[1e-18, 2e-18, 3e-18], [4e-18, 5e-18, 6e-18]]]),
            concentration=np.ones((1, 2)),
            charge=np.zeros((1, 2)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6]),
        )
        fractions = data.mass_fractions
        np.testing.assert_allclose(np.sum(fractions, axis=-1), np.ones((1, 2)))
        assert fractions.shape == (1, 2, 3)

    def test_mass_fractions_zero_mass(self) -> None:
        """Mass fractions are zero when total mass is zero."""
        data = ParticleData(
            masses=np.zeros((2, 2, 2)),
            concentration=np.ones((2, 2)),
            charge=np.zeros((2, 2)),
            density=np.array([1000.0, 1200.0]),
            volume=np.ones(2),
        )
        np.testing.assert_array_equal(data.mass_fractions, np.zeros((2, 2, 2)))
        assert data.mass_fractions.shape == (2, 2, 2)


class TestParticleDataCopy:
    """Tests for ParticleData copy method."""

    def test_copy_creates_independent_arrays(self) -> None:
        """copy() returns new arrays that do not share memory."""
        original = ParticleData(
            masses=np.ones((1, 3, 2)),
            concentration=np.ones((1, 3)),
            charge=np.zeros((1, 3)),
            density=np.array([1000.0, 1200.0]),
            volume=np.array([1e-6]),
        )
        copied = original.copy()

        # Mutate each field in the original to verify independence
        original.masses[0, 0, 0] = 10.0
        original.concentration[0, 0] = 5.0
        original.charge[0, 0] = 3.0
        original.density[0] = 2000.0
        original.volume[0] = 2e-6

        # Verify copied arrays are unchanged
        assert copied.masses[0, 0, 0] == 1.0
        assert copied.concentration[0, 0] == 1.0
        assert copied.charge[0, 0] == 0.0
        assert copied.density[0] == 1000.0
        assert copied.volume[0] == 1e-6

        # Verify no shared memory for all arrays
        assert not np.shares_memory(copied.masses, original.masses)
        assert not np.shares_memory(
            copied.concentration, original.concentration
        )
        assert not np.shares_memory(copied.charge, original.charge)
        assert not np.shares_memory(copied.density, original.density)
        assert not np.shares_memory(copied.volume, original.volume)

    def test_copy_preserves_values(self) -> None:
        """copy() preserves all values."""
        rng = np.random.default_rng(seed=42)
        original = ParticleData(
            masses=rng.random((2, 4, 3)),
            concentration=rng.random((2, 4)),
            charge=rng.integers(-5, 5, (2, 4)).astype(float),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6, 2e-6]),
        )
        copied = original.copy()

        np.testing.assert_array_equal(copied.masses, original.masses)
        np.testing.assert_array_equal(
            copied.concentration, original.concentration
        )
        np.testing.assert_array_equal(copied.charge, original.charge)
        np.testing.assert_array_equal(copied.density, original.density)
        np.testing.assert_array_equal(copied.volume, original.volume)
