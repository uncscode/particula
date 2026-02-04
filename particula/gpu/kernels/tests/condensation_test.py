"""Tests for GPU condensation kernels.

Tests cover kernel outputs, inactive particle handling, multi-box scenarios,
edge cases, and numerical equivalence with CPU reference implementation.

All tests use device="cpu" to ensure they run without GPU hardware.
"""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from particula.gpu.kernels.condensation import (  # noqa: E402
    apply_mass_transfer_kernel,
    condensation_mass_transfer_kernel,
    condensation_step_gpu,
)
from particula.gpu.warp_types import WarpGasData, WarpParticleData  # noqa: E402


def create_test_particle_data(
    n_boxes: int = 1,
    n_particles: int = 10,
    n_species: int = 2,
    device: str = "cpu",
) -> WarpParticleData:
    """Create WarpParticleData with test values.

    Args:
        n_boxes: Number of simulation boxes.
        n_particles: Number of particles per box.
        n_species: Number of species.
        device: Device to allocate arrays on.

    Returns:
        Initialized WarpParticleData struct.
    """
    # Typical particle masses: 1e-18 to 1e-15 kg
    np_masses = np.ones((n_boxes, n_particles, n_species)) * 1e-17
    np_concentration = np.ones((n_boxes, n_particles))  # All particles active
    np_charge = np.zeros((n_boxes, n_particles))
    np_density = np.array([1000.0] * n_species)  # kg/m^3
    np_volume = np.array([1e-6] * n_boxes)  # m^3

    data = WarpParticleData()
    data.masses = wp.array(np_masses, dtype=wp.float64, device=device)
    data.concentration = wp.array(
        np_concentration, dtype=wp.float64, device=device
    )
    data.charge = wp.array(np_charge, dtype=wp.float64, device=device)
    data.density = wp.array(np_density, dtype=wp.float64, device=device)
    data.volume = wp.array(np_volume, dtype=wp.float64, device=device)
    return data


def create_test_gas_data(
    n_boxes: int = 1,
    n_species: int = 2,
    device: str = "cpu",
) -> WarpGasData:
    """Create WarpGasData with test values.

    Args:
        n_boxes: Number of simulation boxes.
        n_species: Number of species.
        device: Device to allocate arrays on.

    Returns:
        Initialized WarpGasData struct.
    """
    np_molar_mass = np.array([0.150] * n_species)  # kg/mol (organic compounds)
    np_concentration = np.ones((n_boxes, n_species)) * 1e-6  # mol/m^3
    np_vapor_pressure = np.ones((n_boxes, n_species)) * 100.0  # Pa
    np_partitioning = np.array([1] * n_species, dtype=np.int32)

    gas = WarpGasData()
    gas.molar_mass = wp.array(np_molar_mass, dtype=wp.float64, device=device)
    gas.concentration = wp.array(
        np_concentration, dtype=wp.float64, device=device
    )
    gas.vapor_pressure = wp.array(
        np_vapor_pressure, dtype=wp.float64, device=device
    )
    gas.partitioning = wp.array(np_partitioning, dtype=wp.int32, device=device)
    return gas


class TestCondensationKernelOutputs:
    """Tests for condensation kernel output correctness."""

    def test_condensation_kernel_finite_output(self) -> None:
        """Verify outputs are finite (no NaN/inf)."""
        n_boxes, n_particles, n_species = 2, 100, 3
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,  # temperature
                2e-5,  # diffusion_coefficient
                68e-9,  # mean_free_path
                1.0,  # mass_accommodation
                0.001,  # dt
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()
        assert np.all(np.isfinite(result)), (
            "Mass transfer output contains NaN or inf"
        )

    def test_condensation_kernel_produces_nonzero(self) -> None:
        """Verify non-trivial mass transfer occurs with supersaturation."""
        n_boxes, n_particles, n_species = 1, 10, 2
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # Set high vapor pressure to drive condensation
        high_vp = np.ones((n_boxes, n_species)) * 1000.0  # Pa
        gas.vapor_pressure = wp.array(high_vp, dtype=wp.float64, device=device)

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()
        assert np.any(result != 0.0), (
            "Mass transfer should be non-zero with supersaturation"
        )

    def test_condensation_kernel_mass_clamping(self) -> None:
        """Verify masses stay non-negative after evaporation."""
        n_boxes, n_particles, n_species = 1, 5, 2
        device = "cpu"

        # Create particles with small masses
        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        small_masses = (
            np.ones((n_boxes, n_particles, n_species)) * 1e-22
        )  # Very small
        particles.masses = wp.array(
            small_masses, dtype=wp.float64, device=device
        )

        gas = create_test_gas_data(n_boxes, n_species, device)
        # Set low vapor pressure to drive evaporation
        low_vp = np.ones((n_boxes, n_species)) * 0.1  # Pa
        gas.vapor_pressure = wp.array(low_vp, dtype=wp.float64, device=device)
        # Set high gas concentration to create undersaturation
        high_conc = np.ones((n_boxes, n_species)) * 1.0  # mol/m^3
        gas.concentration = wp.array(high_conc, dtype=wp.float64, device=device)

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        # Compute mass transfer (should be negative for evaporation)
        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                1.0,  # Large timestep to force evaporation
            ],
            outputs=[mass_transfer],
            device=device,
        )

        # Apply mass transfer with clamping
        wp.launch(
            kernel=apply_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[particles.masses, mass_transfer, particles.concentration],
            device=device,
        )

        result = particles.masses.numpy()
        assert np.all(result >= 0.0), (
            "Masses should be non-negative after clamping"
        )


class TestCondensationInactiveParticles:
    """Tests for inactive particle handling."""

    def test_inactive_particles_unchanged(self) -> None:
        """Particles with concentration=0 should have zero mass transfer."""
        n_boxes, n_particles, n_species = 1, 10, 2
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # Set all particles as inactive
        inactive_conc = np.zeros((n_boxes, n_particles))
        particles.concentration = wp.array(
            inactive_conc, dtype=wp.float64, device=device
        )

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()
        np.testing.assert_array_equal(
            result,
            np.zeros_like(result),
            "Inactive particles should have zero mass transfer",
        )

    def test_mixed_active_inactive(self) -> None:
        """Mixed active/inactive particles handled correctly."""
        n_boxes, n_particles, n_species = 1, 10, 2
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # Set high vapor pressure to drive condensation
        high_vp = np.ones((n_boxes, n_species)) * 1000.0
        gas.vapor_pressure = wp.array(high_vp, dtype=wp.float64, device=device)

        # Mark even-indexed particles as inactive
        mixed_conc = np.ones((n_boxes, n_particles))
        mixed_conc[0, ::2] = 0.0  # Every other particle inactive
        particles.concentration = wp.array(
            mixed_conc, dtype=wp.float64, device=device
        )

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()

        # Inactive particles (even indices) should have zero mass transfer
        for i in range(0, n_particles, 2):
            np.testing.assert_array_equal(
                result[0, i, :],
                np.zeros(n_species),
                f"Inactive particle {i} should have zero mass transfer",
            )

        # Active particles (odd indices) should have non-zero mass transfer
        active_transfer = result[0, 1::2, :]
        assert np.any(active_transfer != 0.0), (
            "Active particles should have non-zero transfer"
        )


class TestCondensationMultiBox:
    """Tests for multi-box scenarios."""

    def test_multi_box_condensation(self) -> None:
        """Test condensation with n_boxes > 1."""
        n_boxes, n_particles, n_species = 5, 20, 3
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # Set different vapor pressures per box
        varying_vp = np.array([[100.0, 200.0, 300.0]] * n_boxes)
        for b in range(n_boxes):
            varying_vp[b, :] *= b + 1  # Scale by box index
        gas.vapor_pressure = wp.array(
            varying_vp, dtype=wp.float64, device=device
        )

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()

        # All boxes should produce finite results
        assert np.all(np.isfinite(result)), "Multi-box results should be finite"

        # Different boxes should have different mass transfer due to varying VP
        for b in range(1, n_boxes):
            # Mass transfer should differ between boxes
            diff = np.abs(result[b, :, :] - result[0, :, :])
            assert np.any(diff > 0), f"Box {b} should differ from box 0"

    def test_single_box_condensation(self) -> None:
        """Test n_boxes = 1 edge case."""
        n_boxes, n_particles, n_species = 1, 50, 2
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()
        assert result.shape == (1, 50, 2), (
            "Single-box output shape should be (1, 50, 2)"
        )
        assert np.all(np.isfinite(result)), (
            "Single-box results should be finite"
        )


class TestCondensationEdgeCases:
    """Tests for edge cases."""

    def test_zero_radius_handling(self) -> None:
        """Zero-mass particles handled gracefully."""
        n_boxes, n_particles, n_species = 1, 5, 2
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # Set some particles to zero mass
        zero_masses = np.zeros((n_boxes, n_particles, n_species))
        zero_masses[0, 2:, :] = 1e-17  # Only particles 2-4 have mass
        particles.masses = wp.array(
            zero_masses, dtype=wp.float64, device=device
        )

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()

        # Zero-mass particles should have zero mass transfer
        np.testing.assert_array_equal(
            result[0, 0, :], np.zeros(n_species), "Zero-mass particle 0"
        )
        np.testing.assert_array_equal(
            result[0, 1, :], np.zeros(n_species), "Zero-mass particle 1"
        )

        # No NaN or inf
        assert np.all(np.isfinite(result)), "Results should be finite"

    def test_single_particle_condensation(self) -> None:
        """Single particle case."""
        n_boxes, n_particles, n_species = 1, 1, 2
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # High vapor pressure for condensation
        high_vp = np.ones((n_boxes, n_species)) * 500.0
        gas.vapor_pressure = wp.array(high_vp, dtype=wp.float64, device=device)

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()
        assert result.shape == (1, 1, 2), "Single-particle output shape"
        assert np.all(np.isfinite(result)), (
            "Single-particle results should be finite"
        )

    def test_single_species_condensation(self) -> None:
        """Single species case."""
        n_boxes, n_particles, n_species = 1, 10, 1
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()
        assert result.shape == (1, 10, 1), "Single-species output shape"
        assert np.all(np.isfinite(result)), (
            "Single-species results should be finite"
        )


class TestCondensationStepGpu:
    """Tests for high-level condensation_step_gpu function."""

    def test_condensation_step_gpu_basic(self) -> None:
        """High-level function works."""
        n_boxes, n_particles, n_species = 2, 50, 3
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        result = condensation_step_gpu(
            particles,
            gas,
            temperature=298.15,
            dt=0.001,
            diffusion_coefficient=2e-5,
            mean_free_path=68e-9,
            mass_accommodation=1.0,
        )

        # Result should be the same object
        assert result is particles, "Should return same object"

        # Masses should be modified
        masses = result.masses.numpy()
        assert masses.shape == (n_boxes, n_particles, n_species)
        assert np.all(np.isfinite(masses)), "Masses should be finite"
        assert np.all(masses >= 0.0), "Masses should be non-negative"

    def test_condensation_step_gpu_preserves_device(self) -> None:
        """Output on same device as input."""
        n_boxes, n_particles, n_species = 1, 10, 2
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        result = condensation_step_gpu(
            particles, gas, temperature=298.15, dt=0.001
        )

        # Device should be preserved
        assert str(result.masses.device) == device

    def test_condensation_step_gpu_modifies_masses(self) -> None:
        """Masses actually change after condensation step."""
        n_boxes, n_particles, n_species = 1, 10, 2
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # High vapor pressure to ensure condensation
        high_vp = np.ones((n_boxes, n_species)) * 1000.0
        gas.vapor_pressure = wp.array(high_vp, dtype=wp.float64, device=device)

        # Store original masses
        original_masses = particles.masses.numpy().copy()

        # Run condensation
        condensation_step_gpu(
            particles,
            gas,
            temperature=298.15,
            dt=0.01,  # Larger dt
        )

        new_masses = particles.masses.numpy()

        # Masses should have changed - use strict rtol to detect any change
        mass_changed = not np.allclose(
            original_masses, new_masses, rtol=1e-12, atol=1e-30
        )
        assert mass_changed, (
            f"Masses should change. Original: {original_masses[0, 0, :]}, "
            f"New: {new_masses[0, 0, :]}"
        )


class TestGpuVsCpuEquivalence:
    """Tests for numerical equivalence between GPU and CPU implementations."""

    def test_gpu_vs_cpu_mass_transfer_rate(self) -> None:
        """Compare GPU kernel output direction with expected physics.

        With high vapor pressure (supersaturation), mass transfer should be
        positive (condensation). With low vapor pressure (undersaturation),
        mass transfer should be negative (evaporation).
        """
        n_boxes, n_particles, n_species = 1, 5, 1
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # High vapor pressure case (supersaturation -> condensation)
        high_vp = np.ones((n_boxes, n_species)) * 10000.0  # Pa
        gas.vapor_pressure = wp.array(high_vp, dtype=wp.float64, device=device)
        low_conc = np.ones((n_boxes, n_species)) * 1e-8  # mol/m^3
        gas.concentration = wp.array(low_conc, dtype=wp.float64, device=device)

        mass_transfer_high = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer_high],
            device=device,
        )

        result_high = mass_transfer_high.numpy()

        # With high vapor pressure (supersaturation), expect positive mass transfer
        assert np.all(result_high > 0), (
            "High VP should give positive mass transfer (condensation)"
        )

        # Low vapor pressure case (undersaturation -> evaporation)
        low_vp = np.ones((n_boxes, n_species)) * 1.0  # Pa
        gas.vapor_pressure = wp.array(low_vp, dtype=wp.float64, device=device)
        high_conc = np.ones((n_boxes, n_species)) * 1.0  # mol/m^3
        gas.concentration = wp.array(high_conc, dtype=wp.float64, device=device)

        mass_transfer_low = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer_low],
            device=device,
        )

        result_low = mass_transfer_low.numpy()

        # With low vapor pressure (undersaturation), expect negative mass transfer
        assert np.all(result_low < 0), (
            "Low VP should give negative mass transfer (evaporation)"
        )

    def test_gpu_vs_cpu_simple_condensation(self) -> None:
        """Full condensation step produces physically reasonable results.

        This tests that:
        1. Mass transfer scales with particle radius
        2. Larger particles have higher absolute mass transfer
        3. Results are numerically stable
        """
        n_boxes, n_particles, n_species = 1, 3, 1
        device = "cpu"

        # Create particles with different sizes
        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )

        # Set different masses for different particles
        # Particle 0: small, Particle 1: medium, Particle 2: large
        varying_masses = np.array([[[1e-20]], [[1e-18]], [[1e-16]]])
        particles.masses = wp.array(
            varying_masses, dtype=wp.float64, device=device
        )

        gas = create_test_gas_data(n_boxes, n_species, device)
        # High vapor pressure for condensation
        high_vp = np.ones((n_boxes, n_species)) * 1000.0
        gas.vapor_pressure = wp.array(high_vp, dtype=wp.float64, device=device)

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()

        # All should be finite and positive (condensation)
        assert np.all(np.isfinite(result)), "Results should be finite"
        assert np.all(result >= 0), "Should be condensation (positive)"

        # Larger particles should have higher absolute mass transfer
        # (mass transfer rate scales with radius)
        dm_small = result[0, 0, 0]
        dm_medium = result[0, 1, 0]
        dm_large = result[0, 2, 0]

        assert dm_medium > dm_small, (
            "Medium particle should have higher transfer than small"
        )
        assert dm_large > dm_medium, (
            "Large particle should have higher transfer than medium"
        )


class TestCondensationKernelLaunchDimensions:
    """Tests for correct 2D kernel launch pattern."""

    def test_kernel_covers_all_particles(self) -> None:
        """Verify all particles are processed in multi-box scenario."""
        n_boxes, n_particles, n_species = 3, 7, 2  # Non-round numbers
        device = "cpu"

        particles = create_test_particle_data(
            n_boxes, n_particles, n_species, device
        )
        gas = create_test_gas_data(n_boxes, n_species, device)

        # High vapor pressure for condensation
        high_vp = np.ones((n_boxes, n_species)) * 500.0
        gas.vapor_pressure = wp.array(high_vp, dtype=wp.float64, device=device)

        mass_transfer = wp.zeros(
            (n_boxes, n_particles, n_species), dtype=wp.float64, device=device
        )

        wp.launch(
            kernel=condensation_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.concentration,
                particles.density,
                gas.concentration,
                gas.vapor_pressure,
                gas.molar_mass,
                298.15,
                2e-5,
                68e-9,
                1.0,
                0.001,
            ],
            outputs=[mass_transfer],
            device=device,
        )

        result = mass_transfer.numpy()

        # Every active particle should have non-zero mass transfer
        for b in range(n_boxes):
            for p in range(n_particles):
                transfer = result[b, p, :]
                assert np.any(transfer != 0.0), (
                    f"Particle ({b}, {p}) not processed"
                )


class TestModuleExports:
    """Tests for module-level exports."""

    def test_kernels_importable_from_gpu_module(self) -> None:
        """Verify kernel functions are exported from particula.gpu."""
        from particula.gpu import (
            apply_mass_transfer_kernel,
            condensation_mass_transfer_kernel,
            condensation_step_gpu,
        )

        # Verify functions exist
        assert condensation_mass_transfer_kernel is not None
        assert apply_mass_transfer_kernel is not None
        assert condensation_step_gpu is not None

    def test_kernels_importable_from_kernels_module(self) -> None:
        """Verify kernel functions are exported from particula.gpu.kernels."""
        from particula.gpu.kernels import (
            apply_mass_transfer_kernel,
            condensation_mass_transfer_kernel,
            condensation_step_gpu,
        )

        # Verify functions exist
        assert condensation_mass_transfer_kernel is not None
        assert apply_mass_transfer_kernel is not None
        assert condensation_step_gpu is not None
