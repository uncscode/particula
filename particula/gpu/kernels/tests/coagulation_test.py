"""End-to-end tests for GPU coagulation kernels."""

# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportIndexIssue=false

from __future__ import annotations

# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

wp = pytest.importorskip("warp")

from particula.dynamics.coagulation.brownian_kernel import (  # noqa: E402
    get_brownian_kernel_via_system_state,
)
from particula.gas.gas_data import GasData  # noqa: E402
from particula.gpu.conversion import (  # noqa: E402
    from_warp_particle_data,
    to_warp_particle_data,
)
from particula.gpu.dynamics.coagulation_funcs import (  # noqa: E402
    brownian_diffusivity_wp,
    brownian_kernel_pair_wp,
    g_collection_term_wp,
    particle_mean_free_path_wp,
)
from particula.gpu.kernels.coagulation import (  # noqa: E402
    _ensure_volume_array,
    _initialize_rng_states,
    _validate_collision_counts,
    _validate_collision_pairs,
    _validate_device_arrays,
    _validate_device_match,
    _validate_particle_arrays,
    _validate_rng_states,
    apply_coagulation_kernel,
    brownian_coagulation_kernel,
    coagulation_step_gpu,
)
from particula.gpu.properties.gas_properties import (  # noqa: E402
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
)
from particula.gpu.properties.particle_properties import (  # noqa: E402
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    knudsen_number_wp,
    mean_thermal_speed_wp,
)

# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
from particula.particles.particle_data import ParticleData  # noqa: E402
from particula.util import constants  # noqa: E402


@pytest.fixture(params=["cpu"] + (["cuda"] if wp.is_cuda_available() else []))
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


def _make_particle_data(
    n_boxes: int,
    n_particles: int,
    n_species: int,
    concentration_scale: float = 1.0,
) -> Any:
    """Create deterministic particle data for coagulation tests."""
    base_masses = np.linspace(1.0e-18, 2.0e-18, n_species, dtype=np.float64)
    masses = np.empty((n_boxes, n_particles, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        for particle_idx in range(n_particles):
            scale = 1.0 + 0.2 * particle_idx + 0.05 * box_idx
            masses[box_idx, particle_idx, :] = base_masses * scale
    concentration = np.full(
        (n_boxes, n_particles),
        concentration_scale,
        dtype=np.float64,
    )
    charge = np.zeros((n_boxes, n_particles), dtype=np.float64)
    density = np.linspace(1000.0, 1400.0, n_species, dtype=np.float64)
    volume = np.full((n_boxes,), 1.0e-6, dtype=np.float64)
    return ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=density,
        volume=volume,
    )


def _make_gas_data(n_boxes: int, n_species: int) -> GasData:
    """Create gas data for Brownian kernel reference calculations."""
    molar_mass = np.linspace(0.018, 0.05, n_species, dtype=np.float64)
    concentration = np.full((n_boxes, n_species), 1.0e-6, dtype=np.float64)
    partitioning = np.ones((n_species,), dtype=bool)
    names = [f"species_{idx}" for idx in range(n_species)]
    return GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=partitioning,
    )


@wp.kernel
# type: ignore[misc]
def _brownian_kernel_matrix_kernel(
    radii: Any,
    masses: Any,
    temperature: Any,
    pressure: Any,
    boltzmann_constant: Any,
    gas_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    kernel_out: Any,
) -> None:
    """Compute Brownian kernel matrix using shared GPU building blocks."""  # type: ignore
    i_idx, j_idx = wp.tid()  # type: ignore[misc]
    dynamic_viscosity = dynamic_viscosity_wp(
        temperature,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature,
        pressure,
        dynamic_viscosity,
        gas_constant,
    )
    knudsen_i = knudsen_number_wp(mean_free_path, radii[i_idx])
    knudsen_j = knudsen_number_wp(mean_free_path, radii[j_idx])
    slip_i = cunningham_slip_correction_wp(knudsen_i)
    slip_j = cunningham_slip_correction_wp(knudsen_j)
    mobility_i = aerodynamic_mobility_wp(
        radii[i_idx],
        slip_i,
        dynamic_viscosity,
    )
    mobility_j = aerodynamic_mobility_wp(
        radii[j_idx],
        slip_j,
        dynamic_viscosity,
    )
    diffusivity_i = brownian_diffusivity_wp(
        temperature,
        mobility_i,
        boltzmann_constant,
    )
    diffusivity_j = brownian_diffusivity_wp(
        temperature,
        mobility_j,
        boltzmann_constant,
    )
    speed_i = mean_thermal_speed_wp(
        masses[i_idx],
        temperature,
        boltzmann_constant,
    )
    speed_j = mean_thermal_speed_wp(
        masses[j_idx],
        temperature,
        boltzmann_constant,
    )
    mean_free_path_i = particle_mean_free_path_wp(diffusivity_i, speed_i)
    mean_free_path_j = particle_mean_free_path_wp(diffusivity_j, speed_j)
    g_term_i = g_collection_term_wp(mean_free_path_i, radii[i_idx])
    g_term_j = g_collection_term_wp(mean_free_path_j, radii[j_idx])
    kernel_out[i_idx, j_idx] = brownian_kernel_pair_wp(
        radii[i_idx],
        radii[j_idx],
        diffusivity_i,
        diffusivity_j,
        g_term_i,
        g_term_j,
        speed_i,
        speed_j,
        wp.float64(1.0),
    )


def test_brownian_kernel_matrix_parity_gpu_cpu(device: str) -> None:
    """GPU Brownian kernel matrix matches CPU reference."""
    temperature = 298.15
    pressure = 101325.0
    radii = np.array([1.0e-8, 5.0e-8, 1.0e-7], dtype=np.float64)
    masses = np.array([1.0e-21, 4.0e-21, 8.0e-21], dtype=np.float64)

    expected = get_brownian_kernel_via_system_state(
        particle_radius=radii,
        particle_mass=masses,
        temperature=temperature,
        pressure=pressure,
    )

    radii_wp = wp.array(radii, dtype=wp.float64, device=device)
    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    kernel_wp = wp.zeros(
        (len(radii), len(radii)), dtype=wp.float64, device=device
    )

    wp.launch(
        _brownian_kernel_matrix_kernel,
        dim=(len(radii), len(radii)),
        inputs=[
            radii_wp,
            masses_wp,
            wp.float64(temperature),
            wp.float64(pressure),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
        ],
        outputs=[kernel_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(kernel_wp.numpy(), expected, rtol=1.0e-7)


def test_coagulation_statistical_collision_rate(device: str) -> None:  # noqa: E1136, E1135
    """Collision counts follow expected Brownian rate statistics."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 0.5
    n_steps = 60
    particles = _make_particle_data(n_boxes=1, n_particles=6, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    total_collisions = 0
    for step_idx in range(n_steps):
        _, _, collision_counts = coagulation_step_gpu(
            gpu_particles,
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
            rng_seed=42 + step_idx,
            max_collisions=16,
        )
        wp.synchronize()
        collision_array = np.asarray(collision_counts.numpy())
        total_collisions += int(collision_array.sum())

    density_value = float(np.asarray(particles.density).ravel().item(0))
    masses_array: np.ndarray = np.asarray(particles.masses, dtype=np.float64)
    masses_slice = np.ravel(masses_array[0])
    radii = np.cbrt(3.0 * masses_slice / (4.0 * np.pi * density_value))
    masses = masses_slice
    kernel_matrix = get_brownian_kernel_via_system_state(
        particle_radius=radii,
        particle_mass=masses,
        temperature=temperature,
        pressure=pressure,
    )
    kernel_matrix_array = np.asarray(kernel_matrix, dtype=np.float64)
    kernel_values = np.asarray(kernel_matrix_array)[
        np.triu_indices(len(radii), k=1)
    ]
    kernel_values = np.atleast_1d(kernel_values)
    volume = float(np.asarray(particles.volume).ravel().item(0))
    expected_mean = np.sum(kernel_values) * time_step * n_steps / volume
    expected_sigma = np.sqrt(expected_mean)
    assert total_collisions == pytest.approx(
        expected_mean, abs=3.0 * expected_sigma
    )


def test_coagulation_multi_box_independence(device: str) -> None:
    """Collision counts remain isolated per box."""
    temperature = 300.0
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=3, n_particles=5, n_species=1)
    particles.concentration[1, :] = 0.0
    particles.concentration[2, 0] = 0.0

    gpu_particles = to_warp_particle_data(particles, device=device)
    _, _, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        rng_seed=12,
        max_collisions=16,
    )
    wp.synchronize()
    result = np.asarray(collision_counts.numpy())

    assert result.reshape(-1)[1] == 0
    assert result.reshape(-1)[0] >= result.reshape(-1)[2]


def test_coagulation_mass_conservation(device: str) -> None:
    """Coagulation conserves total mass in each box."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)

    initial_mass = np.sum(particles.masses)
    coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        rng_seed=7,
        max_collisions=8,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)

    final_mass = np.sum(result.masses)
    npt.assert_allclose(final_mass, initial_mass, rtol=1.0e-12)


def test_coagulation_marks_inactive_particles(device: str) -> None:
    """Merged particles are marked inactive and mass is transferred."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=4, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _, _, collision_counts = coagulation_step_gpu(
        gpu_particles,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        volume=1.0e-18,
        rng_seed=3,
        max_collisions=8,
    )
    result = from_warp_particle_data(gpu_particles, sync=True)

    assert np.asarray(collision_counts.numpy()).sum() > 0
    assert np.any(result.concentration == 0.0)
    assert np.max(result.masses) >= np.max(particles.masses)


def test_brownian_coagulation_kernel_inactive_particles(
    device: str,
) -> None:
    """Brownian kernel returns no collisions when particles are inactive."""
    n_boxes = 1
    n_particles = 2
    n_species = 1
    masses = wp.array(
        np.full((n_boxes, n_particles, n_species), 1.0e-18, dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    concentration = wp.zeros(
        (n_boxes, n_particles),
        dtype=wp.float64,
        device=device,
    )
    density = wp.array(
        np.array([1000.0], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )
    volume = wp.array(
        np.array([1.0e-6], dtype=np.float64),
        dtype=wp.float64,
        device=device,
    )

    radii = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    diffusivities = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    g_terms = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    speeds = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    active_flags = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )
    collision_pairs = wp.zeros((n_boxes, 4, 2), dtype=wp.int32, device=device)
    n_collisions = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    rng_states = wp.zeros((n_boxes,), dtype=wp.uint32, device=device)

    wp.launch(
        brownian_coagulation_kernel,
        dim=(n_boxes,),
        inputs=[
            masses,
            concentration,
            density,
            volume,
            wp.float64(298.15),
            wp.float64(101325.0),
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
            wp.float64(1.0),
            radii,
            diffusivities,
            g_terms,
            speeds,
            active_flags,
            collision_pairs,
            n_collisions,
            rng_states,
        ],
        device=device,
    )
    wp.synchronize()

    assert np.asarray(n_collisions.numpy()).item() == 0
    npt.assert_allclose(np.asarray(radii.numpy()), 0.0)
    npt.assert_allclose(np.asarray(diffusivities.numpy()), 0.0)
    npt.assert_allclose(np.asarray(g_terms.numpy()), 0.0)
    npt.assert_allclose(np.asarray(speeds.numpy()), 0.0)


def test_apply_coagulation_kernel_merges_particles(device: str) -> None:
    """Apply kernel merges masses and zeroes merged particle concentration."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    collision_pairs = np.array([[[0, 1]]], dtype=np.int32)
    n_collisions = np.array([1], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())

    npt.assert_allclose(result_masses[0, 0, 0], 3.0e-18)
    npt.assert_allclose(result_masses[0, 1, 0], 0.0)
    npt.assert_allclose(result_concentration[0, 1], 0.0)

    n_collisions_zero = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            collision_pairs_wp,
            n_collisions_zero,
        ],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(np.asarray(masses_wp.numpy()), result_masses)
    npt.assert_allclose(
        np.asarray(concentration_wp.numpy()), result_concentration
    )


def test_apply_coagulation_kernel_skips_self_pair(device: str) -> None:
    """Apply kernel ignores self-collisions without mutating arrays."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    collision_pairs = np.array([[[0, 0]]], dtype=np.int32)
    n_collisions = np.array([1], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())

    npt.assert_allclose(result_masses, masses)
    npt.assert_allclose(result_concentration, concentration)


def test_apply_coagulation_kernel_skips_empty_pair(device: str) -> None:
    """Apply kernel ignores entries when collision index is out of range."""
    masses = np.array([[[1.0e-18], [2.0e-18]]], dtype=np.float64)
    concentration = np.array([[1.0, 1.0]], dtype=np.float64)
    collision_pairs = np.array([[[0, 1]]], dtype=np.int32)
    n_collisions = np.array([0], dtype=np.int32)

    masses_wp = wp.array(masses, dtype=wp.float64, device=device)
    concentration_wp = wp.array(concentration, dtype=wp.float64, device=device)
    collision_pairs_wp = wp.array(
        collision_pairs, dtype=wp.int32, device=device
    )
    n_collisions_wp = wp.array(n_collisions, dtype=wp.int32, device=device)

    wp.launch(
        apply_coagulation_kernel,
        dim=(1, 1),
        inputs=[
            masses_wp,
            concentration_wp,
            collision_pairs_wp,
            n_collisions_wp,
        ],
        device=device,
    )
    wp.synchronize()

    result_masses = np.asarray(masses_wp.numpy())
    result_concentration = np.asarray(concentration_wp.numpy())

    npt.assert_allclose(result_masses, masses)
    npt.assert_allclose(result_concentration, concentration)


def test_kernels_init_exports() -> None:
    """Kernel module exports coagulation utilities."""
    from particula.gpu import kernels

    assert kernels.brownian_coagulation_kernel is not None
    assert kernels.apply_coagulation_kernel is not None
    assert kernels.coagulation_step_gpu is not None


def test_coagulation_validation_rejects_bad_shapes(device: str) -> None:
    """Validation helpers reject mismatched shapes."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)

    gpu_particles.masses = wp.zeros(
        (1, 2, 3),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle masses shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.concentration = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle concentration shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.volume = wp.zeros(
        (2,),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle volume shape"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_particles.density = wp.zeros(
        (3,),
        dtype=wp.float64,
        device=device,
    )
    with pytest.raises(ValueError, match="particle density length"):
        _validate_particle_arrays(gpu_particles, 1, 2, 2)

    collision_pairs = wp.zeros(
        (1, 3, 2),
        dtype=wp.int32,
        device=device,
    )
    with pytest.raises(ValueError, match="collision_pairs shape"):
        _validate_collision_pairs(
            collision_pairs, (1, 2, 2), gpu_particles.masses.device
        )

    n_collisions = wp.zeros(
        (2,),
        dtype=wp.int32,
        device=device,
    )
    with pytest.raises(ValueError, match="n_collisions shape"):
        _validate_collision_counts(
            n_collisions, (1,), gpu_particles.masses.device
        )

    rng_states = wp.zeros(
        (2,),
        dtype=wp.uint32,
        device=device,
    )
    with pytest.raises(ValueError, match="rng_states shape"):
        _validate_rng_states(rng_states, (1,), gpu_particles.masses.device)


def test_coagulation_validation_rejects_device_mismatch(device: str) -> None:
    """Validation helpers reject device mismatches."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    collision_pairs = wp.zeros(
        (1, 2, 2),
        dtype=wp.int32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="collision_pairs buffer"):
        _validate_collision_pairs(
            collision_pairs, (1, 2, 2), gpu_particles.masses.device
        )

    n_collisions = wp.zeros(
        (1,),
        dtype=wp.int32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="n_collisions"):
        _validate_collision_counts(
            n_collisions, (1,), gpu_particles.masses.device
        )

    rng_states = wp.zeros(
        (1,),
        dtype=wp.uint32,
        device=wrong_device,
    )
    with pytest.raises(ValueError, match="rng_states"):
        _validate_rng_states(rng_states, (1,), gpu_particles.masses.device)


def test_coagulation_validate_device_arrays(device: str) -> None:
    """Device array validation passes for matching devices and fails otherwise."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)

    _validate_device_arrays(gpu_particles, gpu_particles.masses.device)

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA not available for mismatch test")

    gpu_particles.volume = wp.zeros((1,), dtype=wp.float64, device=wrong_device)
    with pytest.raises(ValueError, match="particle volume device mismatch"):
        _validate_device_arrays(gpu_particles, gpu_particles.masses.device)


def test_coagulation_ensure_volume_array(device: str) -> None:
    """Volume helper returns a device array and validates shapes."""
    volume_array = _ensure_volume_array(1.0e-6, n_boxes=2, device=device)
    assert volume_array.shape == (2,)
    npt.assert_allclose(np.asarray(volume_array.numpy()), [1.0e-6, 1.0e-6])

    bad_volume = wp.zeros((3,), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="volume shape does not match"):
        _ensure_volume_array(bad_volume, n_boxes=2, device=device)

    volume_array = _ensure_volume_array(1.5e-6, n_boxes=2, device=device)
    _validate_device_match("volume", volume_array, volume_array.device)


def test_initialize_rng_states_changes_output(device: str) -> None:
    """RNG state initialization writes nonzero data."""
    rng_states = wp.zeros((4,), dtype=wp.uint32, device=device)
    wp.launch(
        _initialize_rng_states,
        dim=4,
        inputs=[wp.uint32(123), rng_states],
        device=device,
    )
    wp.synchronize()

    rng_values = np.asarray(rng_states.numpy())
    assert np.any(rng_values != 0)
