"""Tests for Warp GPU data types.

Tests cover struct instantiation, field shapes, array dtypes,
single-box and multi-box scenarios for WarpParticleData and WarpGasData.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from particula.gpu.warp_types import WarpGasData, WarpParticleData


class TestWarpParticleDataCreation:
    """Tests for WarpParticleData struct instantiation."""

    def test_warp_particle_data_creation(self) -> None:
        """Verify WarpParticleData struct can be instantiated with valid inputs."""
        n_boxes, n_particles, n_species = 2, 100, 3

        data = WarpParticleData()
        data.masses = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64
        )
        data.concentration = wp.ones((n_boxes, n_particles), dtype=wp.float64)
        data.charge = wp.zeros((n_boxes, n_particles), dtype=wp.float64)
        data.density = wp.array([1000.0, 1200.0, 1500.0], dtype=wp.float64)
        data.volume = wp.array([1e-3, 1e-3], dtype=wp.float64)

        # Verify struct is accessible
        assert data.masses is not None
        assert data.concentration is not None
        assert data.charge is not None
        assert data.density is not None
        assert data.volume is not None

    def test_warp_particle_data_field_shapes(self) -> None:
        """Verify all field shapes match expected dimensions."""
        n_boxes, n_particles, n_species = 2, 50, 4

        data = WarpParticleData()
        data.masses = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64
        )
        data.concentration = wp.ones((n_boxes, n_particles), dtype=wp.float64)
        data.charge = wp.zeros((n_boxes, n_particles), dtype=wp.float64)
        data.density = wp.array(
            [1000.0, 1100.0, 1200.0, 1300.0], dtype=wp.float64
        )
        data.volume = wp.array([1e-6, 2e-6], dtype=wp.float64)

        # Verify shapes
        assert data.masses.shape == (n_boxes, n_particles, n_species)
        assert data.concentration.shape == (n_boxes, n_particles)
        assert data.charge.shape == (n_boxes, n_particles)
        assert data.density.shape == (n_species,)
        assert data.volume.shape == (n_boxes,)

    def test_single_box_particle_data(self) -> None:
        """Test WarpParticleData with n_boxes = 1 (single box)."""
        n_boxes, n_particles, n_species = 1, 200, 2

        data = WarpParticleData()
        data.masses = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64
        )
        data.concentration = wp.ones((n_boxes, n_particles), dtype=wp.float64)
        data.charge = wp.zeros((n_boxes, n_particles), dtype=wp.float64)
        data.density = wp.array([1000.0, 1200.0], dtype=wp.float64)
        data.volume = wp.array([1e-6], dtype=wp.float64)

        # Verify single-box shapes
        assert data.masses.shape == (1, 200, 2)
        assert data.concentration.shape == (1, 200)
        assert data.charge.shape == (1, 200)
        assert data.density.shape == (2,)
        assert data.volume.shape == (1,)

    def test_multi_box_particle_data(self) -> None:
        """Test WarpParticleData with n_boxes > 1 (multi-box CFD scenario)."""
        n_boxes, n_particles, n_species = 5, 100, 3

        data = WarpParticleData()
        data.masses = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64
        )
        data.concentration = wp.ones((n_boxes, n_particles), dtype=wp.float64)
        data.charge = wp.zeros((n_boxes, n_particles), dtype=wp.float64)
        data.density = wp.array([1000.0, 1200.0, 800.0], dtype=wp.float64)
        data.volume = wp.array([1e-6] * n_boxes, dtype=wp.float64)

        # Verify multi-box shapes
        assert data.masses.shape == (5, 100, 3)
        assert data.concentration.shape == (5, 100)
        assert data.charge.shape == (5, 100)
        assert data.density.shape == (3,)
        assert data.volume.shape == (5,)

    def test_particle_data_field_dtypes(self) -> None:
        """Verify array dtypes are float64 for all numerical arrays."""
        n_boxes, n_particles, n_species = 2, 10, 2

        data = WarpParticleData()
        data.masses = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64
        )
        data.concentration = wp.ones((n_boxes, n_particles), dtype=wp.float64)
        data.charge = wp.zeros((n_boxes, n_particles), dtype=wp.float64)
        data.density = wp.array([1000.0, 1200.0], dtype=wp.float64)
        data.volume = wp.array([1e-6, 2e-6], dtype=wp.float64)

        # Verify dtypes
        assert data.masses.dtype == wp.float64
        assert data.concentration.dtype == wp.float64
        assert data.charge.dtype == wp.float64
        assert data.density.dtype == wp.float64
        assert data.volume.dtype == wp.float64

    def test_particle_data_with_numpy_values(self) -> None:
        """Test creating WarpParticleData from NumPy arrays."""
        n_boxes, n_particles, n_species = 2, 5, 3

        # Create NumPy arrays
        np_masses = np.random.rand(n_boxes, n_particles, n_species) * 1e-18
        np_concentration = np.ones((n_boxes, n_particles))
        np_charge = np.zeros((n_boxes, n_particles))
        np_density = np.array([1000.0, 1200.0, 800.0])
        np_volume = np.array([1e-6, 2e-6])

        # Create Warp struct with NumPy data
        data = WarpParticleData()
        data.masses = wp.array(np_masses, dtype=wp.float64)
        data.concentration = wp.array(np_concentration, dtype=wp.float64)
        data.charge = wp.array(np_charge, dtype=wp.float64)
        data.density = wp.array(np_density, dtype=wp.float64)
        data.volume = wp.array(np_volume, dtype=wp.float64)

        # Verify shapes match
        assert data.masses.shape == (n_boxes, n_particles, n_species)
        assert data.concentration.shape == (n_boxes, n_particles)
        assert data.charge.shape == (n_boxes, n_particles)
        assert data.density.shape == (n_species,)
        assert data.volume.shape == (n_boxes,)


class TestWarpGasDataCreation:
    """Tests for WarpGasData struct instantiation."""

    def test_warp_gas_data_creation(self) -> None:
        """Verify WarpGasData struct can be instantiated with valid inputs."""
        n_boxes, n_species = 2, 3

        gas = WarpGasData()
        gas.molar_mass = wp.array([0.018, 0.150, 0.200], dtype=wp.float64)
        gas.concentration = wp.zeros((n_boxes, n_species), dtype=wp.float64)
        gas.vapor_pressure = wp.array(
            np.ones((n_boxes, n_species)) * 1000.0, dtype=wp.float64
        )
        gas.partitioning = wp.array([1, 1, 0], dtype=wp.int32)

        # Verify struct is accessible
        assert gas.molar_mass is not None
        assert gas.concentration is not None
        assert gas.vapor_pressure is not None
        assert gas.partitioning is not None

    def test_warp_gas_data_field_shapes(self) -> None:
        """Verify all field shapes match expected dimensions."""
        n_boxes, n_species = 3, 4

        gas = WarpGasData()
        gas.molar_mass = wp.array(
            [0.018, 0.150, 0.200, 0.250], dtype=wp.float64
        )
        gas.concentration = wp.zeros((n_boxes, n_species), dtype=wp.float64)
        gas.vapor_pressure = wp.ones((n_boxes, n_species), dtype=wp.float64)
        gas.partitioning = wp.array([1, 1, 0, 1], dtype=wp.int32)

        # Verify shapes
        assert gas.molar_mass.shape == (n_species,)
        assert gas.concentration.shape == (n_boxes, n_species)
        assert gas.vapor_pressure.shape == (n_boxes, n_species)
        assert gas.partitioning.shape == (n_species,)

    def test_single_box_gas_data(self) -> None:
        """Test WarpGasData with n_boxes = 1 (single box)."""
        n_boxes, n_species = 1, 2

        gas = WarpGasData()
        gas.molar_mass = wp.array([0.018, 0.044], dtype=wp.float64)
        gas.concentration = wp.zeros((n_boxes, n_species), dtype=wp.float64)
        gas.vapor_pressure = wp.ones((n_boxes, n_species), dtype=wp.float64)
        gas.partitioning = wp.array([1, 0], dtype=wp.int32)

        # Verify single-box shapes
        assert gas.molar_mass.shape == (2,)
        assert gas.concentration.shape == (1, 2)
        assert gas.vapor_pressure.shape == (1, 2)
        assert gas.partitioning.shape == (2,)

    def test_multi_box_gas_data(self) -> None:
        """Test WarpGasData with n_boxes > 1 (multi-box CFD scenario)."""
        n_boxes, n_species = 5, 3

        gas = WarpGasData()
        gas.molar_mass = wp.array([0.018, 0.150, 0.200], dtype=wp.float64)
        gas.concentration = wp.zeros((n_boxes, n_species), dtype=wp.float64)
        gas.vapor_pressure = wp.ones((n_boxes, n_species), dtype=wp.float64)
        gas.partitioning = wp.array([1, 1, 0], dtype=wp.int32)

        # Verify multi-box shapes
        assert gas.molar_mass.shape == (3,)
        assert gas.concentration.shape == (5, 3)
        assert gas.vapor_pressure.shape == (5, 3)
        assert gas.partitioning.shape == (3,)

    def test_gas_data_field_dtypes(self) -> None:
        """Verify array dtypes: float64 for numerical, int32 for partitioning."""
        n_boxes, n_species = 2, 2

        gas = WarpGasData()
        gas.molar_mass = wp.array([0.018, 0.150], dtype=wp.float64)
        gas.concentration = wp.zeros((n_boxes, n_species), dtype=wp.float64)
        gas.vapor_pressure = wp.ones((n_boxes, n_species), dtype=wp.float64)
        gas.partitioning = wp.array([1, 0], dtype=wp.int32)

        # Verify dtypes
        assert gas.molar_mass.dtype == wp.float64
        assert gas.concentration.dtype == wp.float64
        assert gas.vapor_pressure.dtype == wp.float64
        assert gas.partitioning.dtype == wp.int32

    def test_gas_data_with_numpy_values(self) -> None:
        """Test creating WarpGasData from NumPy arrays."""
        n_boxes, n_species = 3, 4

        # Create NumPy arrays
        np_molar_mass = np.array([0.018, 0.029, 0.044, 0.150])
        np_concentration = np.random.rand(n_boxes, n_species) * 1e15
        np_vapor_pressure = np.random.rand(n_boxes, n_species) * 1000.0
        np_partitioning = np.array([1, 0, 1, 1], dtype=np.int32)

        # Create Warp struct with NumPy data
        gas = WarpGasData()
        gas.molar_mass = wp.array(np_molar_mass, dtype=wp.float64)
        gas.concentration = wp.array(np_concentration, dtype=wp.float64)
        gas.vapor_pressure = wp.array(np_vapor_pressure, dtype=wp.float64)
        gas.partitioning = wp.array(np_partitioning, dtype=wp.int32)

        # Verify shapes match
        assert gas.molar_mass.shape == (n_species,)
        assert gas.concentration.shape == (n_boxes, n_species)
        assert gas.vapor_pressure.shape == (n_boxes, n_species)
        assert gas.partitioning.shape == (n_species,)


class TestWarpDataFieldAccess:
    """Tests for field access on 1D, 2D, and 3D arrays."""

    def test_particle_data_3d_array_access(self) -> None:
        """Test accessing elements in 3D masses array."""
        n_boxes, n_particles, n_species = 2, 3, 2

        # Create with specific values
        np_masses = (
            np.arange(n_boxes * n_particles * n_species)
            .reshape(n_boxes, n_particles, n_species)
            .astype(np.float64)
        )

        data = WarpParticleData()
        data.masses = wp.array(np_masses, dtype=wp.float64)

        # Verify shape
        assert data.masses.shape == (2, 3, 2)

        # Convert back to numpy and verify values
        masses_np = data.masses.numpy()
        np.testing.assert_array_equal(masses_np, np_masses)

    def test_particle_data_2d_array_access(self) -> None:
        """Test accessing elements in 2D concentration array."""
        n_boxes, n_particles = 3, 4

        np_concentration = (
            np.arange(n_boxes * n_particles)
            .reshape(n_boxes, n_particles)
            .astype(np.float64)
        )

        data = WarpParticleData()
        data.concentration = wp.array(np_concentration, dtype=wp.float64)

        # Verify shape
        assert data.concentration.shape == (3, 4)

        # Convert back and verify
        conc_np = data.concentration.numpy()
        np.testing.assert_array_equal(conc_np, np_concentration)

    def test_particle_data_1d_array_access(self) -> None:
        """Test accessing elements in 1D density and volume arrays."""
        n_species = 3
        n_boxes = 2

        np_density = np.array([1000.0, 1200.0, 800.0])
        np_volume = np.array([1e-6, 2e-6])

        data = WarpParticleData()
        data.density = wp.array(np_density, dtype=wp.float64)
        data.volume = wp.array(np_volume, dtype=wp.float64)

        # Verify shapes
        assert data.density.shape == (n_species,)
        assert data.volume.shape == (n_boxes,)

        # Convert back and verify
        np.testing.assert_array_almost_equal(data.density.numpy(), np_density)
        np.testing.assert_array_almost_equal(data.volume.numpy(), np_volume)

    def test_gas_data_partitioning_int32(self) -> None:
        """Test partitioning array uses int32 values correctly."""
        n_species = 4

        # Create partitioning mask (1=True, 0=False)
        np_partitioning = np.array([1, 0, 1, 0], dtype=np.int32)

        gas = WarpGasData()
        gas.partitioning = wp.array(np_partitioning, dtype=wp.int32)

        # Verify shape and dtype
        assert gas.partitioning.shape == (n_species,)
        assert gas.partitioning.dtype == wp.int32

        # Convert back and verify values
        part_np = gas.partitioning.numpy()
        np.testing.assert_array_equal(part_np, np_partitioning)


class TestWarpAvailability:
    """Tests for WARP_AVAILABLE sentinel in module."""

    def test_warp_available_sentinel(self) -> None:
        """Verify WARP_AVAILABLE is True when warp is imported."""
        from particula.gpu import WARP_AVAILABLE

        assert WARP_AVAILABLE is True

    def test_module_exports(self) -> None:
        """Verify WarpParticleData and WarpGasData are exported."""
        from particula.gpu import WarpGasData, WarpParticleData

        # Verify classes are importable
        assert WarpParticleData is not None
        assert WarpGasData is not None
