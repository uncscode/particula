"""pytest tests for particle_resolved_method.py."""

import time

import numpy as np

# fmt: off
from particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method import (  # noqa: E501  # pylint: disable=line-too-long
    _calculate_probabilities,
    _final_coagulation_state,
    _interpolate_kernel,
    get_particle_resolved_coagulation_step,
    get_particle_resolved_update_step,
)
from scipy.interpolate import RegularGridInterpolator

# fmt: on


def test_interpolate_kernel():
    """Test the interpolate_kernel function."""
    kernel = np.random.rand(10, 10)
    kernel_radius = np.linspace(0, 1, 10)
    interp_func = _interpolate_kernel(kernel, kernel_radius)
    assert isinstance(interp_func, RegularGridInterpolator)


def test_interpolate_kernel_out_of_bounds_returns_zero() -> None:
    """Out-of-bounds interpolation should return zero values."""
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    kernel_radius = np.array([1.0, 2.0], dtype=np.float64)
    interp_func = _interpolate_kernel(kernel, kernel_radius)

    query_points = np.array(
        [
            [0.5, 1.5],
            [2.5, 1.5],
            [1.5, 2.5],
            [0.5, 2.5],
        ],
        dtype=np.float64,
    )
    values = interp_func(query_points)
    np.testing.assert_allclose(values, 0.0)


def test_calculate_probabilities():
    """Test the calculate_probabilities function."""
    kernel_values = np.array([1.0, 2.0], dtype=np.float64)
    time_step = 1.0
    events = 10
    tests = 5
    volume = 100.0
    probabilities = _calculate_probabilities(
        kernel_values, time_step, events, tests, volume
    )
    expected_probabilities = (
        kernel_values * time_step * events / (tests * volume)
    )
    np.testing.assert_array_almost_equal(probabilities, expected_probabilities)


def test_resolve_final_coagulation_state():
    """Test the resolve_final_coagulation_state function."""
    small_indices = np.array([0, 1, 2], dtype=np.int64)
    large_indices = np.array([2, 3, 4], dtype=np.int64)
    particle_radius = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    updated_small_indices, updated_large_indices = _final_coagulation_state(
        small_indices, large_indices, particle_radius
    )
    assert len(updated_small_indices) == len(small_indices)
    assert len(updated_large_indices) == len(large_indices)
    assert np.all(updated_large_indices >= updated_small_indices)


def test_resolve_final_coagulation_state_large():
    """Test the resolve_final_coagulation_state function with a
    large number of particles.
    """
    num_particles = 100_000
    small_indices = np.random.randint(
        0, num_particles, size=num_particles // 2, dtype=np.int64
    )
    large_indices = np.random.randint(
        0, num_particles, size=num_particles // 2, dtype=np.int64
    )
    particle_radius = np.random.uniform(1.0, 10.0, size=num_particles).astype(
        np.float64
    )

    # Introduce duplicate collisions
    small_indices[:10] = large_indices[:10]

    # Time the function
    start_time = time.time()
    updated_small_indices, updated_large_indices = _final_coagulation_state(
        small_indices, large_indices, particle_radius
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    time_per_particle = elapsed_time / num_particles
    current_version_time_per_particle = 1e-05

    # incase we change the implementation and it is slower
    assert time_per_particle <= current_version_time_per_particle

    assert len(updated_small_indices) == len(small_indices)
    assert len(updated_large_indices) == len(large_indices)


def test_particle_resolved_update_step():
    """Test the particle_resolved_update_step function."""
    particle_radius = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    loss = np.zeros_like(particle_radius)
    gain = np.zeros_like(particle_radius)
    small_index = np.array([0], dtype=np.int64)
    large_index = np.array([1], dtype=np.int64)
    updated_radius, updated_loss, updated_gain = (
        get_particle_resolved_update_step(
            particle_radius, loss, gain, small_index, large_index
        )
    )
    assert updated_radius[0] == 0
    assert updated_radius[1] > 2.0
    assert updated_loss[0] == 1.0
    assert updated_gain[1] == 2.0


def test_update_step_duplicate_large_volume_conservation():
    """Test volume conservation with duplicate large indices."""
    particle_radius = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    loss = np.zeros_like(particle_radius)
    gain = np.zeros_like(particle_radius)
    small_index = np.array([0, 1], dtype=np.int64)
    large_index = np.array([3, 3], dtype=np.int64)

    total_volume_before = np.sum(particle_radius**3)
    updated_radius, _updated_loss, _updated_gain = (
        get_particle_resolved_update_step(
            particle_radius, loss, gain, small_index, large_index
        )
    )
    total_volume_after = np.sum(updated_radius**3)

    np.testing.assert_allclose(
        total_volume_after, total_volume_before, rtol=1e-10
    )


def test_update_step_empty_indices_noop():
    """Test empty indices leave arrays unchanged."""
    particle_radius = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    loss = np.zeros_like(particle_radius)
    gain = np.zeros_like(particle_radius)
    small_index = np.array([], dtype=np.int64)
    large_index = np.array([], dtype=np.int64)

    expected_radius = particle_radius.copy()
    expected_loss = loss.copy()
    expected_gain = gain.copy()

    updated_radius, updated_loss, updated_gain = (
        get_particle_resolved_update_step(
            particle_radius, loss, gain, small_index, large_index
        )
    )

    np.testing.assert_allclose(updated_radius, expected_radius, rtol=1e-10)
    np.testing.assert_allclose(updated_loss, expected_loss, rtol=1e-10)
    np.testing.assert_allclose(updated_gain, expected_gain, rtol=1e-10)


def test_particle_resolved_coagulation_step():
    """Test the particle_resolved_coagulation_step function."""
    particle_radius = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    kernel = np.random.rand(10, 10)
    kernel_radius = np.linspace(0, 1, 10)
    volume = 100.0
    time_step = 1.0
    random_generator = np.random.default_rng(seed=42)
    loss_gain_index = get_particle_resolved_coagulation_step(
        particle_radius,
        kernel,
        kernel_radius,
        volume,
        time_step,
        random_generator,
    )
    assert loss_gain_index.shape[1] == 2
    assert np.all(loss_gain_index[:, 0] < loss_gain_index[:, 1])


def test_particle_resolved_coagulation_step_with_kernel_func():
    """Test direct kernel callable path in particle-resolved coagulation."""
    particle_radius = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    kernel = np.zeros((10, 10), dtype=np.float64)
    kernel_radius = np.linspace(0, 5, 10)
    volume = 100.0
    time_step = 1.0
    random_generator = np.random.default_rng(seed=42)
    call_args = []

    def kernel_func(r_small: np.ndarray, r_large: np.ndarray) -> np.ndarray:
        call_args.append((r_small.copy(), r_large.copy()))
        return np.full_like(r_small, 1e-6, dtype=np.float64)

    loss_gain_index = get_particle_resolved_coagulation_step(
        particle_radius,
        kernel,
        kernel_radius,
        volume,
        time_step,
        random_generator,
        kernel_func=kernel_func,
    )
    assert call_args
    assert loss_gain_index.shape[1] == 2


def test_particle_resolved_coagulation_step_empty_array():
    """Test particle_resolved_coagulation_step function with empty array."""
    particle_radius = np.array([], dtype=np.float64)
    kernel = np.random.rand(10, 10)
    kernel_radius = np.linspace(0, 1, 10)
    volume = 100.0
    time_step = 1.0
    random_generator = np.random.default_rng(seed=42)
    loss_gain_index = get_particle_resolved_coagulation_step(
        particle_radius,
        kernel,
        kernel_radius,
        volume,
        time_step,
        random_generator,
    )
    assert loss_gain_index.size == 0


def test_particle_resolved_coagulation_step_duplicate_collisions():
    """Test particle_resolved_coagulation_step with duplicate collisions."""
    particle_radius = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    kernel = np.random.rand(10, 10)
    kernel_radius = np.linspace(0, 1, 10)
    volume = 100.0
    time_step = 1.0
    random_generator = np.random.default_rng(seed=42)
    loss_gain_index = get_particle_resolved_coagulation_step(
        particle_radius,
        kernel,
        kernel_radius,
        volume,
        time_step,
        random_generator,
    )
    assert loss_gain_index.shape[1] == 2
    assert np.all(loss_gain_index[:, 0] < loss_gain_index[:, 1])
    assert np.all(particle_radius[loss_gain_index[:, 0]] == 0)
    assert np.all(particle_radius[loss_gain_index[:, 1]] > 1.0)


def test_particle_resolved_coagulation_step_particles_outside_kernel_range():
    """Test coagulation when particles are outside the kernel_radius range.

    This reproduces the scenario from the Biomass_Burning_Cloud_Interactions
    notebook where particles grow via condensation to sizes larger than the
    kernel bins, causing _bin_particles to fail.
    """
    # Particles much larger than kernel_radius range (simulating cloud droplets
    # that grew via condensation)
    particle_radius = np.array(
        [1e-5, 2e-5, 5e-5, 1e-4], dtype=np.float64
    )  # 10-100 µm
    # Kernel computed for smaller particle range
    kernel = np.random.rand(10, 10)
    kernel_radius = np.linspace(1e-9, 1e-6, 10)  # 1 nm to 1 µm

    volume = 1e-6
    time_step = 1.0
    random_generator = np.random.default_rng(seed=42)

    # This should not raise an error - particles outside range should be
    # handled gracefully
    loss_gain_index = get_particle_resolved_coagulation_step(
        particle_radius,
        kernel,
        kernel_radius,
        volume,
        time_step,
        random_generator,
    )
    assert loss_gain_index.shape[1] == 2


def test_particle_resolved_coagulation_step_mixed_range_particles():
    """Test coagulation with particles both inside and outside kernel range.

    Some particles are within kernel_radius, others are larger (overflow) or
    smaller (underflow).
    """
    # Mix of particles: some in range, some overflow, some underflow
    particle_radius = np.array(
        [1e-10, 1e-9, 5e-8, 1e-7, 1e-6, 1e-5],  # underflow to overflow
        dtype=np.float64,
    )
    kernel = np.random.rand(10, 10)
    kernel_radius = np.linspace(1e-9, 1e-7, 10)  # kernel covers middle range

    volume = 1e-6
    time_step = 1.0
    random_generator = np.random.default_rng(seed=42)

    loss_gain_index = get_particle_resolved_coagulation_step(
        particle_radius,
        kernel,
        kernel_radius,
        volume,
        time_step,
        random_generator,
    )
    assert loss_gain_index.shape[1] == 2
