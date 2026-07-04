"""Tests for Warp GPU data types.

Tests cover struct instantiation, field shapes, array dtypes, and
single-box and multi-box scenarios for WarpParticleData,
WarpGasData, and WarpEnvironmentData.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from particula.gpu.warp_types import (  # noqa: E402
    WarpEnvironmentData,
    WarpGasData,
    WarpParticleData,
)


class TestWarpParticleDataCreation:
    """Tests for WarpParticleData struct instantiation."""

    def test_warp_particle_data_creation(self) -> None:
        """Verify WarpParticleData struct can be instantiated."""
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
        """Verify array dtypes: float64 numerical, int32 partitioning."""
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


class TestWarpEnvironmentDataCreation:
    """Tests for WarpEnvironmentData struct instantiation."""

    def test_warp_environment_data_creation(self) -> None:
        """Verify WarpEnvironmentData struct can be instantiated."""
        environment = WarpEnvironmentData()
        environment.temperature = wp.array([298.15, 300.15], dtype=wp.float64)
        environment.pressure = wp.array([101325.0, 100500.0], dtype=wp.float64)
        environment.saturation_ratio = wp.array(
            [[0.95, 1.05], [0.85, 1.15]],
            dtype=wp.float64,
        )

        assert environment.temperature is not None
        assert environment.pressure is not None
        assert environment.saturation_ratio is not None
        assert hasattr(environment, "temperature")
        assert hasattr(environment, "pressure")
        assert hasattr(environment, "saturation_ratio")

    def test_warp_environment_data_field_shapes(self) -> None:
        """Verify environment field shapes match the CPU schema."""
        n_boxes = 2
        n_species = 3

        environment = WarpEnvironmentData()
        environment.temperature = wp.array([298.15, 299.15], dtype=wp.float64)
        environment.pressure = wp.array([101325.0, 99000.0], dtype=wp.float64)
        environment.saturation_ratio = wp.array(
            [[0.90, 1.00, 1.10], [0.85, 0.95, 1.05]],
            dtype=wp.float64,
        )

        assert environment.temperature.shape == (n_boxes,)
        assert environment.pressure.shape == (n_boxes,)
        assert environment.saturation_ratio.shape == (n_boxes, n_species)

    def test_single_box_environment_data(self) -> None:
        """Verify single-box inputs preserve the leading box axis."""
        environment = WarpEnvironmentData()
        environment.temperature = wp.array([298.15], dtype=wp.float64)
        environment.pressure = wp.array([101325.0], dtype=wp.float64)
        environment.saturation_ratio = wp.array(
            [[0.92, 1.08]],
            dtype=wp.float64,
        )

        assert environment.temperature.shape == (1,)
        assert environment.pressure.shape == (1,)
        assert environment.saturation_ratio.shape == (1, 2)

    def test_multi_box_environment_data(self) -> None:
        """Verify multi-box inputs preserve the box axis."""
        n_boxes = 3
        n_species = 2

        environment = WarpEnvironmentData()
        environment.temperature = wp.array(
            [295.15, 296.15, 297.15],
            dtype=wp.float64,
        )
        environment.pressure = wp.array(
            [101325.0, 100800.0, 100200.0],
            dtype=wp.float64,
        )
        environment.saturation_ratio = wp.array(
            [[0.80, 0.90], [0.95, 1.05], [1.10, 1.20]],
            dtype=wp.float64,
        )

        assert environment.temperature.shape == (n_boxes,)
        assert environment.pressure.shape == (n_boxes,)
        assert environment.saturation_ratio.shape == (n_boxes, n_species)

    def test_environment_data_field_dtypes(self) -> None:
        """Verify all environment arrays use float64."""
        environment = WarpEnvironmentData()
        environment.temperature = wp.array([298.15, 301.15], dtype=wp.float64)
        environment.pressure = wp.array([101325.0, 101000.0], dtype=wp.float64)
        environment.saturation_ratio = wp.array(
            [[0.88, 0.98], [1.02, 1.12]],
            dtype=wp.float64,
        )

        assert environment.temperature.dtype == wp.float64
        assert environment.pressure.dtype == wp.float64
        assert environment.saturation_ratio.dtype == wp.float64

    def test_environment_data_with_numpy_values_round_trip(self) -> None:
        """Verify deterministic NumPy inputs survive Warp round-trip access."""
        temperature = np.array([298.15, 301.25], dtype=np.float64)
        pressure = np.array([101325.0, 99500.0], dtype=np.float64)
        saturation_ratio = np.array(
            [[0.91, 1.00, 1.07], [0.85, 0.97, 1.11]],
            dtype=np.float64,
        )

        environment = WarpEnvironmentData()
        environment.temperature = wp.array(temperature, dtype=wp.float64)
        environment.pressure = wp.array(pressure, dtype=wp.float64)
        environment.saturation_ratio = wp.array(
            saturation_ratio,
            dtype=wp.float64,
        )

        assert environment.temperature.shape == (2,)
        assert environment.pressure.shape == (2,)
        assert environment.saturation_ratio.shape == (2, 3)
        np.testing.assert_array_equal(
            environment.temperature.numpy(), temperature
        )
        np.testing.assert_array_equal(environment.pressure.numpy(), pressure)
        np.testing.assert_array_equal(
            environment.saturation_ratio.numpy(),
            saturation_ratio,
        )


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

    def test_check_warp_available_returns_true_when_warp_imports(self) -> None:
        """Verify the package helper reports True when Warp imports."""
        from particula import gpu as gpu_package

        assert gpu_package._check_warp_available() is True

    def test_check_warp_available_returns_false_on_import_error(self) -> None:
        """Verify the package helper reports False on Warp import failure."""
        import builtins
        from unittest.mock import patch

        from particula import gpu as gpu_package

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "warp":
                raise ImportError("No module named 'warp'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            assert gpu_package._check_warp_available() is False

    def test_warp_available_sentinel(self) -> None:
        """Verify WARP_AVAILABLE is True when warp is imported."""
        from particula.gpu import WARP_AVAILABLE

        assert WARP_AVAILABLE is True

    def test_module_exports(self) -> None:
        """Verify environment, gas, and particle exports are available."""
        from particula.gpu import (
            WarpEnvironmentData,
            WarpGasData,
            WarpParticleData,
            from_warp_environment_data,
            to_warp_environment_data,
        )

        # Verify classes are importable
        assert WarpParticleData is not None
        assert WarpGasData is not None
        assert WarpEnvironmentData is not None
        assert to_warp_environment_data is not None
        assert from_warp_environment_data is not None
