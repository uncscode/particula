"""Tests for GPU conversion functions.

Tests cover to_warp_particle_data() and to_warp_gas_data() conversion functions
including basic transfer, shape preservation, value integrity, copy behavior,
device selection, multi-box scenarios, and error handling.
"""

import builtins
import importlib
import sys

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from particula.gpu.conversion import (  # noqa: E402
    from_warp_environment_data,
    from_warp_gas_data,
    from_warp_particle_data,
    gpu_context,
    to_warp_environment_data,
    to_warp_gas_data,
    to_warp_particle_data,
)
from particula.gpu.tests.cuda_availability import warp_devices  # noqa: E402


def _assert_environment_gpu_mirror_matches(source, gpu_data) -> None:
    """Assert Warp mirror shapes, values, and dtypes match the CPU source."""
    assert gpu_data.temperature.shape == source.temperature.shape
    assert gpu_data.pressure.shape == source.pressure.shape
    assert gpu_data.saturation_ratio.shape == source.saturation_ratio.shape

    np.testing.assert_array_equal(
        gpu_data.temperature.numpy(),
        source.temperature,
    )
    np.testing.assert_array_equal(
        gpu_data.pressure.numpy(),
        source.pressure,
    )
    np.testing.assert_array_equal(
        gpu_data.saturation_ratio.numpy(),
        source.saturation_ratio,
    )

    assert gpu_data.temperature.dtype == wp.float64
    assert gpu_data.pressure.dtype == wp.float64
    assert gpu_data.saturation_ratio.dtype == wp.float64


def _assert_environment_round_trip_matches(source, restored) -> None:
    """Assert CPU→Warp→CPU preserves environment shapes and values exactly."""
    assert restored.temperature.shape == source.temperature.shape
    assert restored.pressure.shape == source.pressure.shape
    assert restored.saturation_ratio.shape == source.saturation_ratio.shape

    np.testing.assert_array_equal(restored.temperature, source.temperature)
    np.testing.assert_array_equal(restored.pressure, source.pressure)
    np.testing.assert_array_equal(
        restored.saturation_ratio,
        source.saturation_ratio,
    )


def _assert_gas_gpu_mirror_matches(
    source,
    gpu_data,
    vapor_pressure: np.ndarray,
) -> None:
    """Assert Warp gas mirror preserves audited shapes, values, and dtypes."""
    assert gpu_data.molar_mass.shape == source.molar_mass.shape
    assert gpu_data.concentration.shape == source.concentration.shape
    assert gpu_data.vapor_pressure.shape == vapor_pressure.shape
    assert gpu_data.partitioning.shape == source.partitioning.shape

    np.testing.assert_array_equal(
        gpu_data.molar_mass.numpy(), source.molar_mass
    )
    np.testing.assert_array_equal(
        gpu_data.concentration.numpy(),
        source.concentration,
    )
    np.testing.assert_array_equal(
        gpu_data.vapor_pressure.numpy(),
        vapor_pressure,
    )
    np.testing.assert_array_equal(
        gpu_data.partitioning.numpy(),
        source.partitioning.astype(np.int32),
    )

    assert gpu_data.molar_mass.dtype == wp.float64
    assert gpu_data.concentration.dtype == wp.float64
    assert gpu_data.vapor_pressure.dtype == wp.float64
    assert gpu_data.partitioning.dtype == wp.int32


def _assert_gas_round_trip_matches(source, restored) -> None:
    """Assert CPU→Warp→CPU preserves the GasData-owned schema exactly."""
    assert restored.name == source.name
    assert restored.molar_mass.shape == source.molar_mass.shape
    assert restored.concentration.shape == source.concentration.shape
    assert restored.partitioning.shape == source.partitioning.shape

    np.testing.assert_array_equal(restored.molar_mass, source.molar_mass)
    np.testing.assert_array_equal(
        restored.concentration,
        source.concentration,
    )
    np.testing.assert_array_equal(
        restored.partitioning,
        source.partitioning,
    )

    assert restored.molar_mass.dtype == np.float64
    assert restored.concentration.dtype == np.float64
    assert restored.partitioning.dtype == np.bool_


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


@pytest.fixture
def sample_environment_data_single_box():
    """Create single-box EnvironmentData for testing."""
    from particula.gas.environment_data import EnvironmentData

    return EnvironmentData(
        temperature=np.array([298.15]),
        pressure=np.array([101325.0]),
        saturation_ratio=np.array([[0.95, 1.05]]),
    )


@pytest.fixture
def sample_environment_data_multi_box():
    """Create multi-box EnvironmentData for testing."""
    from particula.gas.environment_data import EnvironmentData

    return EnvironmentData(
        temperature=np.array([298.15, 305.0]),
        pressure=np.array([101325.0, 90000.0]),
        saturation_ratio=np.array(
            [[0.95, 1.05, 0.75], [0.8, 0.9, 1.1]],
        ),
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

    def test_to_warp_gas_data_preserves_shapes_and_dtypes(
        self, sample_gas_data
    ) -> None:
        """Test the Warp mirror locks the audited gas schema contract."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        expected_vapor_pressure = np.zeros(
            sample_gas_data.concentration.shape,
            dtype=np.float64,
        )

        _assert_gas_gpu_mirror_matches(
            sample_gas_data,
            gpu_data,
            expected_vapor_pressure,
        )
        assert not hasattr(gpu_data, "name")

    def test_to_warp_gas_data_converts_partitioning_bool_to_int32(
        self, sample_gas_data
    ) -> None:
        """Test bool partitioning becomes the GPU int32 contract."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        partitioning_np = gpu_data.partitioning.numpy()
        expected = np.array([1, 1, 0], dtype=np.int32)

        np.testing.assert_array_equal(partitioning_np, expected)
        assert partitioning_np.dtype == np.int32

    def test_to_warp_gas_data_preserves_explicit_vapor_pressure_values(
        self, sample_gas_data
    ) -> None:
        """Test explicit vapor pressure stays on the GPU mirror unchanged."""
        vp = np.array([[1000.0, 500.0, 200.0], [1100.0, 550.0, 220.0]])

        gpu_data = to_warp_gas_data(
            sample_gas_data, device="cpu", vapor_pressure=vp
        )

        _assert_gas_gpu_mirror_matches(
            sample_gas_data,
            gpu_data,
            vp,
        )

    def test_to_warp_gas_data_defaults_vapor_pressure_to_zeros(
        self, sample_gas_data
    ) -> None:
        """Test omitted vapor pressure defaults to zero-filled GPU storage."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        expected = np.zeros(
            sample_gas_data.concentration.shape, dtype=np.float64
        )
        np.testing.assert_array_equal(gpu_data.vapor_pressure.numpy(), expected)
        assert (
            gpu_data.vapor_pressure.shape == sample_gas_data.concentration.shape
        )
        assert gpu_data.vapor_pressure.dtype == wp.float64

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


class TestToWarpEnvironmentData:
    """Tests for to_warp_environment_data() function."""

    def test_package_exports_environment_helpers_when_warp_unavailable(
        self,
        sample_environment_data_single_box,
        monkeypatch,
    ) -> None:
        """Test package-level helper imports stay available without Warp."""
        from particula import gpu as gpu_package

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "warp":
                raise ImportError("No module named 'warp'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        module_name = "_particula_gpu_without_warp"
        spec = importlib.util.spec_from_file_location(
            module_name,
            gpu_package.__file__,
        )
        assert spec is not None
        assert spec.loader is not None

        unavailable_gpu = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unavailable_gpu)

        assert unavailable_gpu.WARP_AVAILABLE is False
        assert unavailable_gpu.to_warp_environment_data is not None
        assert unavailable_gpu.from_warp_environment_data is not None

        with pytest.raises(RuntimeError, match="Warp is not installed"):
            unavailable_gpu.to_warp_environment_data(
                sample_environment_data_single_box,
                device="cpu",
            )

    def test_environment_data_warp_unavailable_error(
        self, sample_environment_data_single_box
    ) -> None:
        """Test helper propagates Warp-unavailable RuntimeError."""
        from unittest.mock import patch

        with patch(
            "particula.gpu.conversion._ensure_warp_available",
            side_effect=RuntimeError("Warp is not installed"),
        ):
            with pytest.raises(RuntimeError, match="Warp is not installed"):
                to_warp_environment_data(
                    sample_environment_data_single_box,
                    device="cpu",
                )

    def test_to_warp_environment_data_default(
        self, sample_environment_data_single_box
    ) -> None:
        """Test helper returns populated environment fields."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_single_box, device="cpu"
        )

        assert gpu_data.temperature is not None
        assert gpu_data.pressure is not None
        assert gpu_data.saturation_ratio is not None

    def test_environment_data_single_box_shapes_and_values(
        self, sample_environment_data_single_box
    ) -> None:
        """Test single-box conversion preserves shapes, values, and dtypes."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_single_box, device="cpu"
        )
        _assert_environment_gpu_mirror_matches(
            sample_environment_data_single_box,
            gpu_data,
        )

    def test_environment_data_multi_box_shapes_and_values(
        self, sample_environment_data_multi_box
    ) -> None:
        """Test multi-box conversion preserves shapes, values, and dtypes."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_multi_box, device="cpu"
        )
        _assert_environment_gpu_mirror_matches(
            sample_environment_data_multi_box,
            gpu_data,
        )

    def test_environment_data_invalid_device_error(
        self, sample_environment_data_single_box
    ) -> None:
        """Test invalid device raises shared RuntimeError style."""
        with pytest.raises(RuntimeError) as exc_info:
            to_warp_environment_data(
                sample_environment_data_single_box,
                device="invalid_device_xyz",
            )

        error_msg = str(exc_info.value)
        assert "invalid_device_xyz" in error_msg
        assert "not found" in error_msg

    def test_environment_data_copy_true_independence(
        self, sample_environment_data_single_box
    ) -> None:
        """Test copy=True creates independent CPU-backed Warp arrays."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_single_box,
            device="cpu",
            copy=True,
        )

        original_temperature = (
            sample_environment_data_single_box.temperature.copy()
        )
        original_pressure = sample_environment_data_single_box.pressure.copy()
        original_saturation_ratio = (
            sample_environment_data_single_box.saturation_ratio.copy()
        )

        sample_environment_data_single_box.temperature[0] = 310.0
        sample_environment_data_single_box.pressure[0] = 95000.0
        sample_environment_data_single_box.saturation_ratio[0, 0] = 0.5

        np.testing.assert_array_equal(
            gpu_data.temperature.numpy(), original_temperature
        )
        np.testing.assert_array_equal(
            gpu_data.pressure.numpy(), original_pressure
        )
        np.testing.assert_array_equal(
            gpu_data.saturation_ratio.numpy(), original_saturation_ratio
        )

    def test_environment_data_copy_false_cpu_behavior(
        self, sample_environment_data_single_box
    ) -> None:
        """Test copy=False does not expose a live CPU view after conversion."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_single_box,
            device="cpu",
            copy=False,
        )

        np.testing.assert_array_equal(
            gpu_data.temperature.numpy(),
            sample_environment_data_single_box.temperature,
        )
        np.testing.assert_array_equal(
            gpu_data.pressure.numpy(),
            sample_environment_data_single_box.pressure,
        )
        np.testing.assert_array_equal(
            gpu_data.saturation_ratio.numpy(),
            sample_environment_data_single_box.saturation_ratio,
        )

        original_temperature = (
            sample_environment_data_single_box.temperature.copy()
        )
        original_pressure = sample_environment_data_single_box.pressure.copy()
        original_saturation_ratio = (
            sample_environment_data_single_box.saturation_ratio.copy()
        )

        sample_environment_data_single_box.temperature[0] = 302.0
        sample_environment_data_single_box.pressure[0] = 98000.0
        sample_environment_data_single_box.saturation_ratio[0, 1] = 0.85

        gpu_temperature = gpu_data.temperature.numpy()
        gpu_pressure = gpu_data.pressure.numpy()
        gpu_saturation_ratio = gpu_data.saturation_ratio.numpy()

        np.testing.assert_array_equal(gpu_temperature, original_temperature)
        assert not np.array_equal(
            gpu_temperature,
            sample_environment_data_single_box.temperature,
        )

        np.testing.assert_array_equal(gpu_pressure, original_pressure)
        assert not np.array_equal(
            gpu_pressure,
            sample_environment_data_single_box.pressure,
        )

        np.testing.assert_array_equal(
            gpu_saturation_ratio, original_saturation_ratio
        )
        assert not np.array_equal(
            gpu_saturation_ratio,
            sample_environment_data_single_box.saturation_ratio,
        )

    def test_environment_data_copy_false_uses_wp_from_numpy_copy_false(
        self, sample_environment_data_single_box, monkeypatch
    ) -> None:
        """Test copy=False forwards copy=False to wp.from_numpy."""
        calls: list[dict[str, object]] = []
        import types

        class FakeWarpEnvironmentData:
            pass

        fake_wp = types.SimpleNamespace()
        fake_wp.float64 = "float64"

        def fake_from_numpy(values, **kwargs):
            calls.append(kwargs.copy())
            return types.SimpleNamespace(
                numpy=lambda: np.asarray(values),
                dtype=kwargs["dtype"],
            )

        fake_wp.from_numpy = fake_from_numpy

        monkeypatch.setattr(
            "particula.gpu.conversion._ensure_warp_available",
            lambda: fake_wp,
        )
        monkeypatch.setattr(
            "particula.gpu.conversion._validate_device",
            lambda _wp, _device: None,
        )
        monkeypatch.setitem(
            sys.modules,
            "particula.gpu.warp_types",
            types.SimpleNamespace(WarpEnvironmentData=FakeWarpEnvironmentData),
        )

        gpu_data = to_warp_environment_data(
            sample_environment_data_single_box,
            device="cpu",
            copy=False,
        )

        assert len(calls) == 3
        assert all(call.get("copy") is False for call in calls)
        np.testing.assert_array_equal(
            gpu_data.temperature.numpy(),
            sample_environment_data_single_box.temperature,
        )
        np.testing.assert_array_equal(
            gpu_data.pressure.numpy(),
            sample_environment_data_single_box.pressure,
        )
        np.testing.assert_array_equal(
            gpu_data.saturation_ratio.numpy(),
            sample_environment_data_single_box.saturation_ratio,
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

    def test_to_warp_gas_data_raises_value_error_for_vapor_pressure_shape_mismatch(
        self, sample_gas_data
    ) -> None:
        """Test vapor pressure shape failures report actual and expected dims."""
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

    def test_from_warp_gas_data_restores_supplied_names(
        self, sample_gas_data
    ) -> None:
        """Test supplied CPU names survive the round-trip exactly."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data, name=sample_gas_data.name)

        _assert_gas_round_trip_matches(sample_gas_data, result)

    def test_from_warp_gas_data_generates_placeholder_names_when_name_missing(
        self, sample_gas_data
    ) -> None:
        """Test omitted names produce the documented placeholder names."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data)

        expected_names = ["species_0", "species_1", "species_2"]
        assert result.name == expected_names
        assert result.concentration.shape == sample_gas_data.concentration.shape

    def test_from_warp_gas_data_treats_name_none_as_placeholder_path(
        self, sample_gas_data
    ) -> None:
        """Test explicit name=None follows the same placeholder path."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data, name=None)

        assert result.name == ["species_0", "species_1", "species_2"]

    def test_from_warp_gas_data_converts_partitioning_int32_to_bool(
        self, sample_gas_data
    ) -> None:
        """Test GPU int32 partitioning is restored as NumPy bool values."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data, name=sample_gas_data.name)

        assert result.partitioning.dtype == np.bool_
        np.testing.assert_array_equal(
            result.partitioning, sample_gas_data.partitioning
        )

    def test_from_warp_gas_data_drops_gpu_only_vapor_pressure(
        self, sample_gas_data
    ) -> None:
        """Test GPU-only vapor pressure is present on GPU and lost on restore."""
        vapor_pressure = np.array(
            [[1000.0, 500.0, 200.0], [1100.0, 550.0, 220.0]],
            dtype=np.float64,
        )
        gpu_data = to_warp_gas_data(
            sample_gas_data,
            device="cpu",
            vapor_pressure=vapor_pressure,
        )

        np.testing.assert_array_equal(
            gpu_data.vapor_pressure.numpy(),
            vapor_pressure,
        )

        result = from_warp_gas_data(gpu_data, name=sample_gas_data.name)

        _assert_gas_round_trip_matches(sample_gas_data, result)
        assert not hasattr(result, "vapor_pressure")
        assert set(result.__dict__) == {
            "name",
            "molar_mass",
            "concentration",
            "partitioning",
        }

    def test_from_warp_gas_data_preserves_multi_box_round_trip(
        self, sample_gas_data
    ) -> None:
        """Test the leading n_boxes dimension survives the full round-trip."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")
        result = from_warp_gas_data(gpu_data, name=sample_gas_data.name)

        assert result.concentration.shape == (
            sample_gas_data.n_boxes,
            sample_gas_data.n_species,
        )
        np.testing.assert_array_equal(
            result.concentration,
            sample_gas_data.concentration,
        )

    def test_from_warp_gas_data_raises_value_error_for_name_length_mismatch(
        self, sample_gas_data
    ) -> None:
        """Test wrong name lengths fail with actual and expected counts."""
        gpu_data = to_warp_gas_data(sample_gas_data, device="cpu")

        with pytest.raises(ValueError) as exc_info:
            from_warp_gas_data(gpu_data, name=["Only", "Two"])  # Need 3

        error_msg = str(exc_info.value)
        assert "2" in error_msg  # actual length
        assert "3" in error_msg  # expected length

    def test_from_warp_gas_data_raises_value_error_for_empty_name_list(
        self, sample_gas_data
    ) -> None:
        """Test an empty provided name list is rejected as a mismatch."""
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


class TestFromWarpEnvironmentData:
    """Tests for from_warp_environment_data() function."""

    def test_from_warp_environment_data_basic_single_box(
        self, sample_environment_data_single_box
    ) -> None:
        """Test basic GPU-to-CPU transfer returns EnvironmentData."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_single_box,
            device="cpu",
        )
        result = from_warp_environment_data(gpu_data)

        from particula.gas.environment_data import EnvironmentData

        assert isinstance(result, EnvironmentData)

    def test_environment_data_round_trip_single_box(
        self, sample_environment_data_single_box
    ) -> None:
        """Test single-box environment data survives CPU→Warp→CPU."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_single_box,
            device="cpu",
        )
        result = from_warp_environment_data(gpu_data)
        _assert_environment_round_trip_matches(
            sample_environment_data_single_box,
            result,
        )

    def test_environment_data_round_trip_multi_box(
        self, sample_environment_data_multi_box
    ) -> None:
        """Test multi-box environment data survives CPU→Warp→CPU."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_multi_box,
            device="cpu",
        )
        result = from_warp_environment_data(gpu_data)
        _assert_environment_round_trip_matches(
            sample_environment_data_multi_box,
            result,
        )

    @pytest.mark.parametrize("device", warp_devices(wp))
    def test_environment_data_round_trip_available_warp_devices(
        self,
        sample_environment_data_multi_box,
        device: str,
    ) -> None:
        """Test explicit environment round-trip parity on available devices."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_multi_box,
            device=device,
        )
        _assert_environment_gpu_mirror_matches(
            sample_environment_data_multi_box,
            gpu_data,
        )
        result = from_warp_environment_data(gpu_data)
        _assert_environment_round_trip_matches(
            sample_environment_data_multi_box,
            result,
        )

    def test_from_warp_environment_data_sync_true(
        self, sample_environment_data_single_box
    ) -> None:
        """Test default synchronized transfer path."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_single_box,
            device="cpu",
        )
        result = from_warp_environment_data(gpu_data, sync=True)
        _assert_environment_round_trip_matches(
            sample_environment_data_single_box,
            result,
        )

    def test_from_warp_environment_data_sync_false_manual(
        self, sample_environment_data_multi_box
    ) -> None:
        """Test manual synchronization before sync=False transfer."""
        gpu_data = to_warp_environment_data(
            sample_environment_data_multi_box,
            device="cpu",
        )
        wp.synchronize()
        result = from_warp_environment_data(gpu_data, sync=False)
        _assert_environment_round_trip_matches(
            sample_environment_data_multi_box,
            result,
        )

    def test_from_warp_environment_data_invalid_saturation_ratio_shape_raises_value_error(
        self,
    ) -> None:
        """Test malformed Warp environment data fails CPU schema validation."""
        from particula.gpu.warp_types import WarpEnvironmentData

        gpu_data = WarpEnvironmentData()
        gpu_data.temperature = wp.array([298.15], dtype=wp.float64)
        gpu_data.pressure = wp.array([101325.0], dtype=wp.float64)
        gpu_data.saturation_ratio = wp.array([0.95, 1.05], dtype=wp.float64)

        with pytest.raises(ValueError, match="saturation_ratio must be 2D"):
            from_warp_environment_data(gpu_data)


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
        """Test gpu_context works when an explicit portable device is used."""
        # Keep this fast test portable across environments by using Warp cpu.
        with gpu_context(sample_particle_data, device="cpu") as gpu_data:
            assert gpu_data is not None
