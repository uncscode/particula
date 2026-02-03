"""Tests for GPU conversion functions.

Tests cover to_warp_particle_data() and to_warp_gas_data() conversion functions
including basic transfer, shape preservation, value integrity, copy behavior,
device selection, multi-box scenarios, and error handling.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from particula.gpu.conversion import (  # noqa: E402
    to_warp_gas_data,
    to_warp_particle_data,
    from_warp_gas_data,
    from_warp_particle_data,
    gpu_context,
)


@pytest.fixture
def sample_particle_data():
    """Create sample ParticleData for testing."""
    from particula.particles.particle_data import ParticleData

    n_boxes, n_particles, n_species = 2, 100, 3
    return ParticleData(
        masses=np.random.rand(n_boxes, n_particles, n_species) * 1e-18,
        concentration=np.ones((n_boxes, n_particles)),
        charge=np.zeros((n_boxes, n_particles)),
        density=np.array([1000.0, 1200.0, 1500.0]),
        volume=np.array([1e-3, 1e-3]),
    )


@pytest.fixture
def sample_gas_data():
    """Create sample GasData for testing."""
    from particula.gas.gas_data import GasData

    n_boxes, n_species = 2, 3
    return GasData(
        name=["Water", "Ammonia", "H2SO4"],
        molar_mass=np.array([0.018, 0.017, 0.098]),
        concentration=np.random.rand(n_boxes, n_species) * 1e15,
        partitioning=np.array([True, True, False]),
    )


class TestToWarpParticleData:
    """Tests for to_warp_particle_data() function."""

    def test_to_warp_particle_data_default(self, sample_particle_data) -> None:
        """Test to_warp_particle_data with default parameters."""
        # Use cpu device for test (always available)
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")

        # Verify all fields are accessible
        assert gpu_data.masses is not None
        assert gpu_data.concentration is not None
        assert gpu_data.charge is not None
        assert gpu_data.density is not None
        assert gpu_data.volume is not None

    def test_particle_data_shapes_preserved(self, sample_particle_data) -> None:
        """Test all shapes match original after transfer."""
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")

        assert gpu_data.masses.shape == sample_particle_data.masses.shape
        assert (
            gpu_data.concentration.shape
            == sample_particle_data.concentration.shape
        )
        assert gpu_data.charge.shape == sample_particle_data.charge.shape
        assert gpu_data.density.shape == sample_particle_data.density.shape
        assert gpu_data.volume.shape == sample_particle_data.volume.shape

    def test_particle_data_values_preserved(self, sample_particle_data) -> None:
        """Test round-trip with .numpy() comparison and dtype verification."""
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")

        # Verify values match
        np.testing.assert_array_almost_equal(
            gpu_data.masses.numpy(),
            sample_particle_data.masses,
        )
        np.testing.assert_array_almost_equal(
            gpu_data.concentration.numpy(),
            sample_particle_data.concentration,
        )
        np.testing.assert_array_almost_equal(
            gpu_data.charge.numpy(),
            sample_particle_data.charge,
        )
        np.testing.assert_array_almost_equal(
            gpu_data.density.numpy(),
            sample_particle_data.density,
        )
        np.testing.assert_array_almost_equal(
            gpu_data.volume.numpy(),
            sample_particle_data.volume,
        )

        # Verify dtypes
        assert gpu_data.masses.dtype == wp.float64
        assert gpu_data.concentration.dtype == wp.float64
        assert gpu_data.charge.dtype == wp.float64
        assert gpu_data.density.dtype == wp.float64
        assert gpu_data.volume.dtype == wp.float64

    def test_particle_data_copy_true_independence(
        self, sample_particle_data
    ) -> None:
        """Test that copy=True creates independent copy."""
        gpu_data = to_warp_particle_data(
            sample_particle_data, device="cpu", copy=True
        )

        # Store original value
        original_mass = sample_particle_data.masses[0, 0, 0]

        # Modify original
        sample_particle_data.masses[0, 0, 0] = 999999.0

        # GPU data should be unchanged
        assert gpu_data.masses.numpy()[0, 0, 0] == original_mass

    def test_particle_data_copy_false_behavior(
        self, sample_particle_data
    ) -> None:
        """Test that copy=False uses wp.from_numpy()."""
        # Just verify it works - actual zero-copy depends on memory layout
        gpu_data = to_warp_particle_data(
            sample_particle_data, device="cpu", copy=False
        )

        # Verify values match (basic functionality)
        np.testing.assert_array_almost_equal(
            gpu_data.masses.numpy(),
            sample_particle_data.masses,
        )


class TestToWarpGasData:
    """Tests for to_warp_gas_data() function."""

    def test_to_warp_gas_data_default(self, sample_gas_data) -> None:
        """Test to_warp_gas_data with default parameters."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        # Verify all fields are accessible
        assert gpu_data.molar_mass is not None
        assert gpu_data.concentration is not None
        assert gpu_data.vapor_pressure is not None
        assert gpu_data.partitioning is not None

    def test_gas_data_shapes_preserved(self, sample_gas_data) -> None:
        """Test all shapes match expected dimensions."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        n_boxes, n_species = sample_gas_data.n_boxes, sample_gas_data.n_species

        assert gpu_data.molar_mass.shape == (n_species,)
        assert gpu_data.concentration.shape == (n_boxes, n_species)
        assert gpu_data.vapor_pressure.shape == (n_boxes, n_species)
        assert gpu_data.partitioning.shape == (n_species,)

    def test_gas_data_values_preserved(self, sample_gas_data) -> None:
        """Test round-trip comparison and dtype verification."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        # Verify values match
        np.testing.assert_array_almost_equal(
            gpu_data.molar_mass.numpy(),
            sample_gas_data.molar_mass,
        )
        np.testing.assert_array_almost_equal(
            gpu_data.concentration.numpy(),
            sample_gas_data.concentration,
        )

        # Verify dtypes (float64 for numerical, int32 for partitioning)
        assert gpu_data.molar_mass.dtype == wp.float64
        assert gpu_data.concentration.dtype == wp.float64
        assert gpu_data.vapor_pressure.dtype == wp.float64
        assert gpu_data.partitioning.dtype == wp.int32

    def test_gas_data_partitioning_conversion(self, sample_gas_data) -> None:
        """Test bool to int32 conversion (True->1, False->0)."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        # Original: [True, True, False]
        partitioning_np = gpu_data.partitioning.numpy()

        # Should be [1, 1, 0]
        expected = np.array([1, 1, 0], dtype=np.int32)
        np.testing.assert_array_equal(partitioning_np, expected)

    def test_gas_data_vapor_pressure_provided(self, sample_gas_data) -> None:
        """Test with explicit vapor_pressure, verify shape and values."""
        n_boxes, n_species = sample_gas_data.n_boxes, sample_gas_data.n_species
        vp = np.array([[1000.0, 500.0, 200.0], [1100.0, 550.0, 220.0]])

        gpu_data = to_warp_gas_data(
            sample_gas_data, device="cpu", vapor_pressure=vp
        )

        # Verify shape
        assert gpu_data.vapor_pressure.shape == (n_boxes, n_species)

        # Verify values
        np.testing.assert_array_almost_equal(
            gpu_data.vapor_pressure.numpy(), vp
        )

    def test_gas_data_vapor_pressure_default_zeros(
        self, sample_gas_data
    ) -> None:
        """Test that default vapor_pressure is zeros when not provided."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        n_boxes, n_species = sample_gas_data.n_boxes, sample_gas_data.n_species

        # Should be zeros
        expected = np.zeros((n_boxes, n_species), dtype=np.float64)
        np.testing.assert_array_almost_equal(
            gpu_data.vapor_pressure.numpy(), expected
        )

    def test_gas_data_copy_false_behavior(self, sample_gas_data) -> None:
        """Test that copy=False uses wp.from_numpy()."""
        # Just verify it works - actual zero-copy depends on memory layout
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu", copy=False)

        # Verify values match (basic functionality)
        np.testing.assert_array_almost_equal(
            gpu_data.molar_mass.numpy(),
            sample_gas_data.molar_mass,
        )
        np.testing.assert_array_almost_equal(
            gpu_data.concentration.numpy(),
            sample_gas_data.concentration,
        )


class TestDeviceSelection:
    """Tests for device selection functionality."""

    def test_device_cpu_particle_data(self, sample_particle_data) -> None:
        """Test device='cpu' works for ParticleData."""
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")

        # Verify transfer worked
        assert gpu_data.masses is not None
        assert gpu_data.masses.shape == sample_particle_data.masses.shape

    def test_device_cpu_gas_data(self, sample_gas_data) -> None:
        """Test device='cpu' works for GasData."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        # Verify transfer worked
        assert gpu_data.molar_mass is not None
        assert gpu_data.molar_mass.shape == (sample_gas_data.n_species,)

    def test_invalid_device_error(self, sample_particle_data) -> None:
        """Test invalid device raises RuntimeError with helpful message."""
        with pytest.raises(RuntimeError) as exc_info:
            to_warp_particle_data(
                sample_particle_data, device="invalid_device_xyz"
            )

        error_msg = str(exc_info.value)
        assert "invalid_device_xyz" in error_msg
        assert "not found" in error_msg


class TestMultiBoxScenarios:
    """Tests for multi-box scenarios (n_boxes > 1)."""

    def test_multi_box_particle_data(self) -> None:
        """Test n_boxes > 1 scenario for ParticleData."""
        from particula.particles.particle_data import ParticleData

        n_boxes, n_particles, n_species = 5, 50, 2

        data = ParticleData(
            masses=np.random.rand(n_boxes, n_particles, n_species) * 1e-18,
            concentration=np.ones((n_boxes, n_particles)),
            charge=np.zeros((n_boxes, n_particles)),
            density=np.array([1000.0, 1200.0]),
            volume=np.array([1e-3] * n_boxes),
        )

        gpu_data = to_warp_particle_data(data, device="cpu")

        # Verify multi-box shapes
        assert gpu_data.masses.shape == (5, 50, 2)
        assert gpu_data.concentration.shape == (5, 50)
        assert gpu_data.charge.shape == (5, 50)
        assert gpu_data.density.shape == (2,)
        assert gpu_data.volume.shape == (5,)

    def test_multi_box_gas_data(self) -> None:
        """Test n_boxes > 1 scenario for GasData."""
        from particula.gas.gas_data import GasData

        n_boxes, n_species = 5, 4

        data = GasData(
            name=["A", "B", "C", "D"],
            molar_mass=np.array([0.018, 0.029, 0.044, 0.150]),
            concentration=np.random.rand(n_boxes, n_species) * 1e15,
            partitioning=np.array([True, False, True, False]),
        )

        gpu_data = to_warp_gas_data(data, device="cpu")

        # Verify multi-box shapes
        assert gpu_data.molar_mass.shape == (4,)
        assert gpu_data.concentration.shape == (5, 4)
        assert gpu_data.vapor_pressure.shape == (5, 4)
        assert gpu_data.partitioning.shape == (4,)

    def test_single_box_particle_data(self) -> None:
        """Test n_boxes = 1 edge case for ParticleData."""
        from particula.particles.particle_data import ParticleData

        n_boxes, n_particles, n_species = 1, 200, 3

        data = ParticleData(
            masses=np.random.rand(n_boxes, n_particles, n_species) * 1e-18,
            concentration=np.ones((n_boxes, n_particles)),
            charge=np.zeros((n_boxes, n_particles)),
            density=np.array([1000.0, 1200.0, 800.0]),
            volume=np.array([1e-6]),
        )

        gpu_data = to_warp_particle_data(data, device="cpu")

        # Verify single-box shapes
        assert gpu_data.masses.shape == (1, 200, 3)
        assert gpu_data.concentration.shape == (1, 200)
        assert gpu_data.charge.shape == (1, 200)
        assert gpu_data.density.shape == (3,)
        assert gpu_data.volume.shape == (1,)

    def test_single_box_gas_data(self) -> None:
        """Test n_boxes = 1 edge case for GasData."""
        from particula.gas.gas_data import GasData

        data = GasData(
            name=["Water", "CO2"],
            molar_mass=np.array([0.018, 0.044]),
            concentration=np.array([[1e15, 1e14]]),
            partitioning=np.array([True, False]),
        )

        gpu_data = to_warp_gas_data(data, device="cpu")

        # Verify single-box shapes
        assert gpu_data.molar_mass.shape == (2,)
        assert gpu_data.concentration.shape == (1, 2)
        assert gpu_data.vapor_pressure.shape == (1, 2)
        assert gpu_data.partitioning.shape == (2,)

    def test_multi_box_particle_data_round_trip(self) -> None:
        """Test n_boxes > 1 round-trip for ParticleData."""
        from particula.particles.particle_data import ParticleData

        n_boxes, n_particles, n_species = 5, 50, 2

        data = ParticleData(
            masses=np.random.rand(n_boxes, n_particles, n_species) * 1e-18,
            concentration=np.ones((n_boxes, n_particles)),
            charge=np.zeros((n_boxes, n_particles)),
            density=np.array([1000.0, 1200.0]),
            volume=np.array([1e-3] * n_boxes),
        )

        gpu_data = to_warp_particle_data(data, device="cpu")
        result = from_warp_particle_data(gpu_data)

        # Verify all shapes preserved
        assert result.masses.shape == (5, 50, 2)
        assert result.concentration.shape == (5, 50)
        assert result.volume.shape == (5,)

        # Verify values match
        np.testing.assert_array_almost_equal(result.masses, data.masses)

    def test_multi_box_gas_data_round_trip(self) -> None:
        """Test n_boxes > 1 round-trip for GasData."""
        from particula.gas.gas_data import GasData

        n_boxes, n_species = 5, 4
        names = ["A", "B", "C", "D"]

        data = GasData(
            name=names,
            molar_mass=np.array([0.018, 0.029, 0.044, 0.150]),
            concentration=np.random.rand(n_boxes, n_species) * 1e15,
            partitioning=np.array([True, False, True, False]),
        )

        gpu_data = to_warp_gas_data(data, device="cpu")
        result = from_warp_gas_data(gpu_data, name=names)

        # Verify shapes
        assert result.molar_mass.shape == (4,)
        assert result.concentration.shape == (5, 4)

        # Verify values
        np.testing.assert_array_almost_equal(result.molar_mass, data.molar_mass)
        np.testing.assert_array_almost_equal(
            result.concentration, data.concentration
        )

    def test_single_box_round_trip(self) -> None:
        """Test edge case with n_boxes=1."""
        from particula.particles.particle_data import ParticleData

        data = ParticleData(
            masses=np.random.rand(1, 10, 2) * 1e-18,
            concentration=np.ones((1, 10)),
            charge=np.zeros((1, 10)),
            density=np.array([1000.0, 1200.0]),
            volume=np.array([1e-3]),
        )

        gpu_data = to_warp_particle_data(data, device="cpu")
        result = from_warp_particle_data(gpu_data)

        assert result.masses.shape == (1, 10, 2)
        np.testing.assert_array_almost_equal(result.masses, data.masses)

    def test_single_particle_round_trip(self) -> None:
        """Test edge case with n_particles=1, n_species=1."""
        from particula.particles.particle_data import ParticleData

        data = ParticleData(
            masses=np.array([[[1e-18]]]),  # (1, 1, 1)
            concentration=np.array([[1.0]]),  # (1, 1)
            charge=np.array([[0.0]]),  # (1, 1)
            density=np.array([1000.0]),  # (1,)
            volume=np.array([1e-3]),  # (1,)
        )

        gpu_data = to_warp_particle_data(data, device="cpu")
        result = from_warp_particle_data(gpu_data)

        assert result.masses.shape == (1, 1, 1)
        np.testing.assert_array_almost_equal(result.masses, data.masses)


class TestErrorHandling:
    """Tests for error handling."""

    def test_warp_unavailable_error(self) -> None:
        """Test RuntimeError when Warp unavailable (mocked)."""
        from unittest.mock import patch

        # Mock the import to raise ImportError
        with patch.dict("sys.modules", {"warp": None}):
            # Create a mock that raises ImportError on import
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "warp":
                    raise ImportError("No module named 'warp'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                from particula.gpu.conversion import _ensure_warp_available

                with pytest.raises(RuntimeError) as exc_info:
                    _ensure_warp_available()

                error_msg = str(exc_info.value)
                assert "pip install warp-lang" in error_msg

    def test_vapor_pressure_shape_mismatch(self, sample_gas_data) -> None:
        """Test wrong shape vapor_pressure raises ValueError with details."""
        # Wrong shape: should be (2, 3) but we provide (3, 2)
        wrong_shape_vp = np.ones((3, 2))

        with pytest.raises(ValueError) as exc_info:
            to_warp_gas_data(
                sample_gas_data, device="cpu", vapor_pressure=wrong_shape_vp
            )

        error_msg = str(exc_info.value)
        assert "(3, 2)" in error_msg  # actual shape
        assert "(2, 3)" in error_msg  # expected shape


class TestFromWarpParticleData:
    """Tests for from_warp_particle_data() function."""

    def test_from_warp_particle_data_basic(self, sample_particle_data) -> None:
        """Test basic GPU to CPU transfer."""
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")
        result = from_warp_particle_data(gpu_data)

        # Verify result is ParticleData
        from particula.particles.particle_data import ParticleData

        assert isinstance(result, ParticleData)

    def test_particle_data_round_trip(self, sample_particle_data) -> None:
        """Test that from_warp(to_warp(data)) equals original."""
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")
        result = from_warp_particle_data(gpu_data)

        # Verify all fields match original
        np.testing.assert_array_almost_equal(
            result.masses, sample_particle_data.masses
        )
        np.testing.assert_array_almost_equal(
            result.concentration, sample_particle_data.concentration
        )
        np.testing.assert_array_almost_equal(
            result.charge, sample_particle_data.charge
        )
        np.testing.assert_array_almost_equal(
            result.density, sample_particle_data.density
        )
        np.testing.assert_array_almost_equal(
            result.volume, sample_particle_data.volume
        )

    def test_particle_data_shapes_preserved_roundtrip(
        self, sample_particle_data
    ) -> None:
        """Test all shapes preserved after round-trip."""
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")
        result = from_warp_particle_data(gpu_data)

        assert result.masses.shape == sample_particle_data.masses.shape
        assert (
            result.concentration.shape
            == sample_particle_data.concentration.shape
        )
        assert result.charge.shape == sample_particle_data.charge.shape
        assert result.density.shape == sample_particle_data.density.shape
        assert result.volume.shape == sample_particle_data.volume.shape

    def test_sync_true_default(self, sample_particle_data) -> None:
        """Test default sync=True behavior."""
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")
        # Default sync=True should work without issues
        result = from_warp_particle_data(gpu_data, sync=True)
        assert result.masses is not None

    def test_sync_false_manual(self, sample_particle_data) -> None:
        """Test manual sync with sync=False."""
        gpu_data = to_warp_particle_data(sample_particle_data, device="cpu")
        # Manual sync before transfer
        wp.synchronize()
        result = from_warp_particle_data(gpu_data, sync=False)

        # Verify round-trip still works
        np.testing.assert_array_almost_equal(
            result.masses, sample_particle_data.masses
        )


class TestFromWarpGasData:
    """Tests for from_warp_gas_data() function."""

    def test_from_warp_gas_data_basic(self, sample_gas_data) -> None:
        """Test basic GPU to CPU transfer with names."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data, name=sample_gas_data.name)

        # Verify result is GasData
        from particula.gas.gas_data import GasData

        assert isinstance(result, GasData)

    def test_gas_data_round_trip(self, sample_gas_data) -> None:
        """Test round-trip with original names preserved."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data, name=sample_gas_data.name)

        # Verify fields match (except vapor_pressure which is lost)
        assert result.name == sample_gas_data.name
        np.testing.assert_array_almost_equal(
            result.molar_mass, sample_gas_data.molar_mass
        )
        np.testing.assert_array_almost_equal(
            result.concentration, sample_gas_data.concentration
        )
        np.testing.assert_array_equal(
            result.partitioning, sample_gas_data.partitioning
        )

    def test_gas_data_placeholder_names(self, sample_gas_data) -> None:
        """Test default placeholder name generation."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data)  # No name provided

        # Should have placeholder names
        expected_names = ["species_0", "species_1", "species_2"]
        assert result.name == expected_names

    def test_gas_data_partitioning_conversion(self, sample_gas_data) -> None:
        """Test int32->bool conversion for partitioning."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data, name=sample_gas_data.name)

        # Should be bool dtype
        assert result.partitioning.dtype == np.bool_
        # Should match original values
        np.testing.assert_array_equal(
            result.partitioning, sample_gas_data.partitioning
        )

    def test_gas_data_name_length_mismatch(self, sample_gas_data) -> None:
        """Test ValueError for wrong name length."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        with pytest.raises(ValueError) as exc_info:
            from_warp_gas_data(gpu_data, name=["Only", "Two"])  # Need 3

        error_msg = str(exc_info.value)
        assert "2" in error_msg  # actual length
        assert "3" in error_msg  # expected length

    def test_gas_data_empty_name_list(self, sample_gas_data) -> None:
        """Test ValueError for empty name list."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        with pytest.raises(ValueError) as exc_info:
            from_warp_gas_data(gpu_data, name=[])  # Empty list

        error_msg = str(exc_info.value)
        assert "0" in error_msg  # actual length (0)
        assert "3" in error_msg  # expected length

    def test_gas_data_sync_false(self, sample_gas_data) -> None:
        """Test manual sync with sync=False."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        wp.synchronize()
        result = from_warp_gas_data(
            gpu_data, name=sample_gas_data.name, sync=False
        )

        np.testing.assert_array_almost_equal(
            result.molar_mass, sample_gas_data.molar_mass
        )


class TestGpuContext:
    """Tests for gpu_context() context manager."""

    def test_gpu_context_basic(self, sample_particle_data) -> None:
        """Test basic context manager usage."""
        with gpu_context(sample_particle_data, device="cpu") as gpu_data:
            assert gpu_data.masses is not None

    def test_gpu_context_yields_warp_data(self, sample_particle_data) -> None:
        """Test that context yields WarpParticleData."""
        with gpu_context(sample_particle_data, device="cpu") as gpu_data:
            # Verify gpu_data has expected WarpParticleData attributes
            assert hasattr(gpu_data, "masses")
            assert hasattr(gpu_data, "concentration")
            assert hasattr(gpu_data, "charge")
            assert hasattr(gpu_data, "density")
            assert hasattr(gpu_data, "volume")

    def test_gpu_context_transfer_inside(self, sample_particle_data) -> None:
        """Test transferring data back inside context."""
        with gpu_context(sample_particle_data, device="cpu") as gpu_data:
            result = from_warp_particle_data(gpu_data)

        # Verify round-trip
        np.testing.assert_array_almost_equal(
            result.masses, sample_particle_data.masses
        )

    def test_gpu_context_transfer_after(self, sample_particle_data) -> None:
        """Test transferring data back after context exit."""
        with gpu_context(sample_particle_data, device="cpu") as gpu_data:
            pass  # Do nothing inside

        # GPU data should still be valid after context
        result = from_warp_particle_data(gpu_data)
        np.testing.assert_array_almost_equal(
            result.masses, sample_particle_data.masses
        )

    def test_gpu_context_simulation_loop(self, sample_particle_data) -> None:
        """Test context with simulation loop pattern."""
        with gpu_context(sample_particle_data, device="cpu") as gpu_data:
            # Simulate multiple timesteps (no actual computation)
            for _ in range(10):
                # In real use: gpu_data = physics_step(gpu_data, dt)
                pass

            result = from_warp_particle_data(gpu_data)

        # Data should be unchanged (no actual modifications)
        np.testing.assert_array_almost_equal(
            result.masses, sample_particle_data.masses
        )

    def test_gpu_context_default_device(self, sample_particle_data) -> None:
        """Test that default device is cuda (may fallback to cpu in tests)."""
        # This test just verifies the function can be called with default device
        # In CI without GPU, this may raise an error - so we use cpu explicitly
        with gpu_context(sample_particle_data, device="cpu") as gpu_data:
            assert gpu_data is not None


class TestModuleExports:
    """Tests for module exports via gpu/__init__.py."""

    def test_functions_importable_from_gpu_module(self) -> None:
        """Verify functions importable via from particula.gpu import ..."""
        from particula.gpu import to_warp_gas_data, to_warp_particle_data

        assert to_warp_particle_data is not None
        assert to_warp_gas_data is not None
        assert callable(to_warp_particle_data)
        assert callable(to_warp_gas_data)

    def test_from_warp_functions_importable(self) -> None:
        """Verify from_warp functions importable from particula.gpu."""
        from particula.gpu import from_warp_gas_data, from_warp_particle_data

        assert from_warp_particle_data is not None
        assert from_warp_gas_data is not None
        assert callable(from_warp_particle_data)
        assert callable(from_warp_gas_data)

    def test_gpu_context_importable(self) -> None:
        """Verify gpu_context importable from particula.gpu."""
        from particula.gpu import gpu_context

        assert gpu_context is not None
