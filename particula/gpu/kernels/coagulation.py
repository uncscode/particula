"""GPU Brownian coagulation kernels and orchestration utilities.

This module composes Warp ``@wp.func`` building blocks into an end-to-end
coagulation pipeline. Entry-point validation accepts scalar direct inputs,
explicit ``(n_boxes,)`` Warp arrays, or a ``WarpEnvironmentData`` container.
Those sources are normalized into per-box Warp arrays before volume setup,
RNG initialization, or any Warp launch. The kernels operate on GPU-resident
particle data and produce collision pairs that are applied to merge particle
masses in-place.
"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false

from typing import TYPE_CHECKING, Any, no_type_check

import numpy as np

import particula.util.constants as constants

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import warp as wp
else:  # pragma: no cover - runtime import with helpful error
    try:
        import warp as wp
    except ImportError as exc:
        raise ImportError(
            "Warp is required for GPU coagulation kernels. "
            "Install with: pip install warp-lang"
        ) from exc

from particula.gpu.dynamics.coagulation_funcs import (
    brownian_diffusivity_wp,
    brownian_kernel_pair_wp,
    g_collection_term_wp,
    particle_mean_free_path_wp,
)
from particula.gpu.dynamics.condensation_funcs import (
    particle_radius_from_volume_wp,
)
from particula.gpu.kernels.environment import (
    _broadcast_scalar_array,
    _ensure_environment_arrays,
    _is_supported_warp_float_dtype,
    _is_warp_array_like,
    _validate_positive_finite_array,
)
from particula.gpu.properties.gas_properties import (
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
)
from particula.gpu.properties.particle_properties import (
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    knudsen_number_wp,
    mean_thermal_speed_wp,
)


@no_type_check
@wp.kernel
# type: ignore[misc]
def _initialize_rng_states(seed: Any, rng_states: Any) -> None:
    """Initialize or reset per-box RNG states from a seed.

    Args:
        seed: RNG seed value.
        rng_states: RNG state buffer ``(n_boxes,)`` written in place.
    """  # type: ignore
    box_idx = wp.tid()  # type: ignore[misc]
    rng_states[box_idx] = wp.rand_init(wp.int32(seed), wp.int32(box_idx))


@no_type_check
@wp.kernel
# type: ignore[misc]
def brownian_coagulation_kernel(  # noqa: C901
    masses: Any,
    concentration: Any,
    density: Any,
    volume: Any,
    temperature: Any,
    pressure: Any,
    gas_constant: Any,
    boltzmann_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    time_step: Any,
    radii: Any,
    diffusivities: Any,
    g_terms: Any,
    speeds: Any,
    active_flags: Any,
    collision_pairs: Any,
    n_collisions: Any,
    rng_states: Any,
) -> None:
    """Select stochastic coagulation pairs for each box.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        concentration: Particle concentrations ``(n_boxes, n_particles)``.
        density: Species densities ``(n_species,)``.
        volume: Per-box volumes ``(n_boxes,)``.
        temperature: Per-box gas temperatures ``(n_boxes,)`` [K].
        pressure: Per-box gas pressures ``(n_boxes,)`` [Pa].
        gas_constant: Universal gas constant [J/(mol·K)].
        boltzmann_constant: Boltzmann constant [J/K].
        molecular_weight_air: Molecular weight of air [kg/mol].
        ref_viscosity: Reference viscosity at STP [Pa·s].
        ref_temperature: Reference temperature [K].
        sutherland_constant: Sutherland constant [K].
        time_step: Coagulation time step [s].
        radii: Output particle radii ``(n_boxes, n_particles)``.
        diffusivities: Output diffusivities ``(n_boxes, n_particles)``.
        g_terms: Output collection terms ``(n_boxes, n_particles)``.
        speeds: Output mean thermal speeds ``(n_boxes, n_particles)``.
        active_flags: Output active flags ``(n_boxes, n_particles)``.
        collision_pairs: Output collision indices
            ``(n_boxes, max_collisions, 2)``.
        n_collisions: Output collision counts ``(n_boxes,)``.
        rng_states: Per-box RNG states ``(n_boxes,)`` mutated in place during
            pair selection. Reusing this buffer across calls preserves
            caller-owned persistent state unless it is reset before launch.
    """  # type: ignore
    box_idx = wp.tid()  # type: ignore[misc]

    # One thread per box keeps the pair selection sequential and avoids
    # cross-thread races when writing collision pairs.
    n_particles = masses.shape[1]
    n_species = masses.shape[2]

    temperature_value = temperature[box_idx]
    pressure_value = pressure[box_idx]

    dynamic_viscosity = dynamic_viscosity_wp(
        temperature_value,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )
    mean_free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature_value,
        pressure_value,
        dynamic_viscosity,
        gas_constant,
    )

    active_count = wp.int32(0)
    for particle_idx in range(n_particles):
        if concentration[box_idx, particle_idx] <= wp.float64(0.0):
            active_flags[box_idx, particle_idx] = wp.int32(0)
            radii[box_idx, particle_idx] = wp.float64(0.0)
            diffusivities[box_idx, particle_idx] = wp.float64(0.0)
            g_terms[box_idx, particle_idx] = wp.float64(0.0)
            speeds[box_idx, particle_idx] = wp.float64(0.0)
            continue

        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species_idx in range(n_species):
            species_mass = masses[box_idx, particle_idx, species_idx]
            total_mass += species_mass
            total_volume += species_mass / density[species_idx]

        if total_volume <= wp.float64(0.0) or total_mass <= wp.float64(0.0):
            active_flags[box_idx, particle_idx] = wp.int32(0)
            radii[box_idx, particle_idx] = wp.float64(0.0)
            diffusivities[box_idx, particle_idx] = wp.float64(0.0)
            g_terms[box_idx, particle_idx] = wp.float64(0.0)
            speeds[box_idx, particle_idx] = wp.float64(0.0)
            continue

        radius = particle_radius_from_volume_wp(total_volume)
        knudsen = knudsen_number_wp(mean_free_path, radius)
        slip = cunningham_slip_correction_wp(knudsen)
        mobility = aerodynamic_mobility_wp(radius, slip, dynamic_viscosity)
        diffusivity = brownian_diffusivity_wp(
            temperature_value, mobility, boltzmann_constant
        )
        speed = mean_thermal_speed_wp(
            total_mass, temperature_value, boltzmann_constant
        )
        particle_mean_free_path = particle_mean_free_path_wp(diffusivity, speed)
        g_term = g_collection_term_wp(particle_mean_free_path, radius)

        active_flags[box_idx, particle_idx] = wp.int32(1)
        radii[box_idx, particle_idx] = radius
        diffusivities[box_idx, particle_idx] = diffusivity
        g_terms[box_idx, particle_idx] = g_term
        speeds[box_idx, particle_idx] = speed
        active_count += wp.int32(1)

    if active_count < wp.int32(2):
        n_collisions[box_idx] = wp.int32(0)
        return

    # Find particles with min/max radius to compute a k_max upper bound.
    # The Brownian kernel is maximized at the greatest size disparity
    # (small particle has high diffusivity, large particle has large
    # cross-section), so K(r_min, r_max) bounds all pairwise values.
    # This replaces the previous O(n^2) all-pairs scan with O(n).
    r_min = wp.float64(1.0e30)
    r_max = wp.float64(0.0)
    idx_min = wp.int32(-1)
    idx_max = wp.int32(-1)
    for p_idx in range(n_particles):
        if active_flags[box_idx, p_idx] == wp.int32(0):
            continue
        r_p = radii[box_idx, p_idx]
        if r_p < r_min:
            r_min = r_p
            idx_min = wp.int32(p_idx)
        if r_p > r_max:
            r_max = r_p
            idx_max = wp.int32(p_idx)

    k_max = wp.float64(0.0)
    if idx_min >= wp.int32(0) and idx_max >= wp.int32(0):
        if idx_min == idx_max:
            # All active particles have the same radius; use self-pair
            # kernel which is symmetric, so pick any two active indices.
            for p_idx in range(n_particles):
                if (
                    active_flags[box_idx, p_idx] == wp.int32(1)
                    and wp.int32(p_idx) != idx_min
                ):
                    idx_max = wp.int32(p_idx)
                    break
        k_max = brownian_kernel_pair_wp(
            radii[box_idx, idx_min],
            radii[box_idx, idx_max],
            diffusivities[box_idx, idx_min],
            diffusivities[box_idx, idx_max],
            g_terms[box_idx, idx_min],
            g_terms[box_idx, idx_max],
            speeds[box_idx, idx_min],
            speeds[box_idx, idx_max],
            wp.float64(1.0),
        )

    if k_max <= wp.float64(0.0):
        n_collisions[box_idx] = wp.int32(0)
        return

    possible_pairs = (
        wp.float64(active_count)
        * wp.float64(active_count - 1)
        / (wp.float64(2.0))
    )
    expected_trials = k_max * possible_pairs * time_step / volume[box_idx]
    if expected_trials <= wp.float64(0.0):
        n_collisions[box_idx] = wp.int32(0)
        return

    max_collisions = collision_pairs.shape[1]
    collision_count = wp.int32(0)
    state = rng_states[box_idx]

    tests = wp.int32(expected_trials)
    remainder = expected_trials - wp.float64(tests)
    if wp.randf(state) < remainder:
        tests += wp.int32(1)

    if tests <= wp.int32(0):
        n_collisions[box_idx] = wp.int32(0)
        return

    for _ in range(tests):
        if collision_count >= max_collisions or active_count < wp.int32(2):
            break

        selected_i = wp.int32(-1)
        selected_j = wp.int32(-1)
        for _ in range(n_particles * 2):
            idx_i = wp.randi(state, wp.int32(0), wp.int32(n_particles))
            idx_j = wp.randi(state, wp.int32(0), wp.int32(n_particles))
            if idx_i == idx_j:
                continue
            if active_flags[box_idx, idx_i] == wp.int32(1) and active_flags[
                box_idx, idx_j
            ] == wp.int32(1):
                selected_i = idx_i
                selected_j = idx_j
                break

        if selected_i < wp.int32(0) or selected_j < wp.int32(0):
            break

        if selected_j < selected_i:
            temp_idx = selected_i
            selected_i = selected_j
            selected_j = temp_idx

        kernel_value = brownian_kernel_pair_wp(
            radii[box_idx, selected_i],
            radii[box_idx, selected_j],
            diffusivities[box_idx, selected_i],
            diffusivities[box_idx, selected_j],
            g_terms[box_idx, selected_i],
            g_terms[box_idx, selected_j],
            speeds[box_idx, selected_i],
            speeds[box_idx, selected_j],
            wp.float64(1.0),
        )
        if kernel_value <= wp.float64(0.0):
            continue

        if wp.randf(state) < kernel_value / k_max:
            collision_pairs[box_idx, collision_count, 0] = selected_i
            collision_pairs[box_idx, collision_count, 1] = selected_j
            collision_count += wp.int32(1)
            active_flags[box_idx, selected_i] = wp.int32(0)
            active_flags[box_idx, selected_j] = wp.int32(0)
            active_count -= wp.int32(2)

    n_collisions[box_idx] = collision_count
    rng_states[box_idx] = state


@no_type_check
@wp.kernel
# type: ignore[misc]
def apply_coagulation_kernel(
    masses: Any,
    concentration: Any,
    collision_pairs: Any,
    n_collisions: Any,
) -> None:
    """Apply coagulation collisions by merging particle masses.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        concentration: Particle concentrations ``(n_boxes, n_particles)``.
        collision_pairs: Collision index buffer
            ``(n_boxes, max_collisions, 2)``.
        n_collisions: Collision counts ``(n_boxes,)``.
    """  # type: ignore
    box_idx, collision_idx = wp.tid()  # type: ignore[misc]
    if collision_idx >= n_collisions[box_idx]:
        return

    idx_i = collision_pairs[box_idx, collision_idx, 0]
    idx_j = collision_pairs[box_idx, collision_idx, 1]
    if idx_i == idx_j:
        return

    n_species = masses.shape[2]
    for species_idx in range(n_species):
        masses[box_idx, idx_i, species_idx] = (
            masses[box_idx, idx_i, species_idx]
            + masses[box_idx, idx_j, species_idx]
        )
        masses[box_idx, idx_j, species_idx] = wp.float64(0.0)

    concentration[box_idx, idx_j] = wp.float64(0.0)


def _validate_particle_arrays(
    particles: Any,
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> None:
    """Validate particle array shapes and lengths.

    Args:
        particles: GPU particle data container.
        n_boxes: Expected number of boxes.
        n_particles: Expected number of particles per box.
        n_species: Expected number of species.

    Raises:
        ValueError: If particle arrays do not match expected shapes.
    """
    if particles.masses.shape != (n_boxes, n_particles, n_species):
        raise ValueError("particle masses shape does not match expected")
    if particles.density.shape[0] != n_species:
        raise ValueError("particle density length does not match n_species")
    if particles.concentration.shape != (n_boxes, n_particles):
        raise ValueError(
            "particle concentration shape does not match (n_boxes, n_particles)"
        )
    if particles.volume.shape != (n_boxes,):
        raise ValueError("particle volume shape does not match (n_boxes,)")


def _validate_device_match(name: str, array: Any, expected_device: Any) -> None:
    """Validate that a Warp array is on the expected device.

    Args:
        name: Array label for error messages.
        array: Warp array to validate.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the array is not on the expected device.
    """
    device = getattr(array, "device", None)
    if device is None or str(device) != str(expected_device):
        raise ValueError(f"{name} device mismatch")


def _validate_device_arrays(particles: Any, device: Any) -> None:
    """Validate particle arrays share the same Warp device.

    Args:
        particles: GPU particle data container.
        device: Expected Warp device.

    Raises:
        ValueError: If any particle array is on a different device.
    """
    _validate_device_match("particle masses", particles.masses, device)
    _validate_device_match(
        "particle concentration", particles.concentration, device
    )
    _validate_device_match("particle density", particles.density, device)
    _validate_device_match("particle volume", particles.volume, device)


def _validate_collision_pairs(
    collision_pairs: Any,
    expected_shape: tuple[int, int, int],
    expected_device: Any,
) -> None:
    """Validate collision pair buffer shape and device.

    Args:
        collision_pairs: Collision pairs buffer.
        expected_shape: Expected ``(n_boxes, max_collisions, 2)`` shape.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the buffer shape or device mismatches.
    """
    if collision_pairs.shape != expected_shape:
        raise ValueError(
            f"collision_pairs shape {collision_pairs.shape} does not match "
            f"expected {expected_shape}"
        )
    if collision_pairs.dtype != wp.int32:
        raise ValueError("collision_pairs buffer must use dtype int32")
    _validate_device_match(
        "collision_pairs buffer", collision_pairs, expected_device
    )


def _validate_collision_counts(
    n_collisions: Any,
    expected_shape: tuple[int],
    expected_device: Any,
) -> None:
    """Validate collision count buffer shape and device.

    Args:
        n_collisions: Collision counts buffer.
        expected_shape: Expected ``(n_boxes,)`` shape.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the buffer shape or device mismatches.
    """
    if n_collisions.shape != expected_shape:
        raise ValueError(
            f"n_collisions shape {n_collisions.shape} does not match expected "
            f"{expected_shape}"
        )
    if n_collisions.dtype != wp.int32:
        raise ValueError("n_collisions buffer must use dtype int32")
    _validate_device_match("n_collisions buffer", n_collisions, expected_device)


def _validate_rng_states(
    rng_states: Any,
    expected_shape: tuple[int],
    expected_device: Any,
) -> None:
    """Validate RNG state buffer shape and device.

    Args:
        rng_states: RNG state buffer.
        expected_shape: Expected ``(n_boxes,)`` shape.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the buffer shape or device mismatches.
    """
    if rng_states.shape != expected_shape:
        raise ValueError(
            f"rng_states shape {rng_states.shape} does not match expected "
            f"{expected_shape}"
        )
    if rng_states.dtype != wp.uint32:
        raise ValueError("rng_states buffer must use dtype uint32")
    _validate_device_match("rng_states buffer", rng_states, expected_device)


def _validate_time_step(time_step: float) -> float:
    """Validate the coagulation time step before any mutation.

    Args:
        time_step: Proposed coagulation time step in seconds.

    Returns:
        The validated time step as a Python ``float``.

    Raises:
        ValueError: If ``time_step`` is not a finite nonnegative real value.
    """
    if isinstance(time_step, bool):
        raise ValueError("time_step must be finite and nonnegative")

    try:
        time_step_value = float(time_step)
    except (TypeError, ValueError) as exc:
        raise ValueError("time_step must be finite and nonnegative") from exc

    if not np.isfinite(time_step_value) or time_step_value < 0.0:
        raise ValueError("time_step must be finite and nonnegative")

    return time_step_value


def _validate_max_collisions(max_collisions: int) -> int:
    """Validate the supported collision-buffer length before allocation.

    Args:
        max_collisions: Proposed maximum accepted collisions per box.

    Returns:
        The validated collision limit as a Python ``int``.

    Raises:
        ValueError: If ``max_collisions`` is not a supported positive integer.
    """
    if isinstance(max_collisions, bool):
        raise ValueError(
            "max_collisions must be a positive integer <= 2147483647"
        )

    try:
        max_collisions_value = int(max_collisions)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_collisions must be a positive integer <= 2147483647"
        ) from exc

    if (
        max_collisions_value <= 0
        or max_collisions_value > np.iinfo(np.int32).max
        or max_collisions_value != max_collisions
    ):
        raise ValueError(
            "max_collisions must be a positive integer <= 2147483647"
        )

    return max_collisions_value


def _ensure_volume_array(
    volume: float | Any,
    n_boxes: int,
    device: Any,
) -> Any:
    """Ensure a GPU volume array is available.

    Args:
        volume: Volume scalar or Warp array.
        n_boxes: Number of boxes.
        device: Warp device.

    Returns:
        Warp array of volumes ``(n_boxes,)``.

    Raises:
        ValueError: If volume array has mismatched shape, device, or domain.

    Notes:
        Valid Warp arrays are returned unchanged so launch code can reuse
        caller-owned per-box volume buffers.
    """
    volume_array: Any = volume
    if _is_warp_array_like(volume_array):
        if volume_array.shape != (n_boxes,):
            raise ValueError("volume shape does not match (n_boxes,)")
        _validate_device_match("volume", volume_array, device)
        if not _is_supported_warp_float_dtype(volume_array.dtype):
            raise ValueError("volume must use a supported Warp float dtype")
        _validate_positive_finite_array(
            "volume", volume_array, "coagulation_step_gpu"
        )
        return volume_array

    if hasattr(volume, "shape"):
        raise ValueError("volume must be a Warp array with shape (n_boxes,)")

    if isinstance(volume, bool) or not isinstance(volume, (float, np.floating)):
        raise ValueError("volume must be a floating scalar or Warp array")

    volume_scalar = float(volume)
    if not np.isfinite(volume_scalar) or volume_scalar <= 0.0:
        raise ValueError("volume must be finite and > 0")

    return _broadcast_scalar_array(volume_scalar, n_boxes, device)


def coagulation_step_gpu(
    particles: Any,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float,
    volume: float | Any | None = None,
    max_collisions: int = 256,
    rng_seed: int = 0,
    collision_pairs: Any | None = None,
    n_collisions: Any | None = None,
    rng_states: Any | None = None,
    *,
    initialize_rng: bool = False,
    environment: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Execute one Brownian coagulation timestep on the GPU.

    Direct temperature and pressure inputs are validated and normalized before
    volume setup, RNG initialization, or any Warp kernel launch. Caller-owned
    RNG buffers are only reset when ``initialize_rng=True`` explicitly opts in
    to reinitialization.

    Args:
        particles: GPU-resident particle data.
        temperature: Direct gas temperature as either a scalar or a Warp array
            with shape ``(n_boxes,)``. Use ``None`` only with
            ``environment=...``.
        pressure: Direct gas pressure as either a scalar or a Warp array with
            shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
        time_step: Coagulation time step in seconds.
        volume: Per-box volume [m^3]. If None, uses ``particles.volume``.
        max_collisions: Maximum number of collisions per box.
        rng_seed: Seed used whenever this call initializes Warp RNG states.
            This always applies to the omitted-``rng_states`` convenience path.
            For caller-owned persistent buffers, the seed is only consumed when
            ``initialize_rng=True`` explicitly requests a reset.
        collision_pairs: Optional preallocated collision buffer.
        n_collisions: Optional preallocated collision count buffer.
        rng_states: Optional preallocated RNG state buffer. When omitted, this
            function allocates a call-local ``(n_boxes,)`` buffer, seeds it
            from ``rng_seed``, and uses it only for the current call. When
            provided, the caller owns the persistent GPU-resident sidecar
            buffer and it is reused as-is across repeated calls unless
            ``initialize_rng=True`` explicitly requests a reset from
            ``rng_seed``.
        initialize_rng: Explicit reset flag for caller-provided
            ``rng_states``. The default ``False`` path validates the buffer and
            reuses its existing state without reseeding, even if ``rng_seed``
            matches an earlier call. Set ``True`` to launch
            ``_initialize_rng_states`` after validation and reset the
            persistent buffer from ``rng_seed``. This argument does not affect
            omitted ``rng_states`` convenience allocation, which always
            initializes the internal buffer for the current call.
        environment: Optional ``WarpEnvironmentData`` with ``(n_boxes,)``
            temperature and pressure arrays on the same device as ``particles``.
            This mode is supported when both direct inputs are ``None``.

    Returns:
        Tuple containing ``particles`` after in-place coagulation, the
        ``collision_pairs`` buffer for the step, and the ``n_collisions``
        buffer with per-box accepted collision counts.

    Raises:
        ValueError: If array shapes or devices mismatch expectations.
        ValueError: If direct ``temperature`` or ``pressure`` inputs are mixed
            with ``environment``.
        ValueError: If direct inputs are missing when ``environment`` is
            omitted.
        ValueError: If environment arrays do not match ``(n_boxes,)`` or the
            caller device.

    Notes:
        Accepted environment sources are scalar direct inputs, direct
        ``(n_boxes,)`` Warp arrays, hybrid scalar-plus-Warp-array direct
        inputs, or keyword-only ``environment=...`` execution.

        ``initialize_rng`` and ``environment`` remain keyword-only so existing
        positional scalar callers stay source-compatible.

        Validation runs before volume normalization, RNG setup, and kernel
        launches so invalid ``time_step``, shape, dtype, or device
        combinations fail without mutating particle state or allocating
        downstream launch work. The normalized environment arrays are
        forwarded directly into the launch path.

        Supported RNG setup cases are:

        - Omitted ``rng_states``: allocate a call-local internal buffer and
          initialize it from ``rng_seed`` for this call.
        - Provided ``rng_states`` with ``initialize_rng=False``: validate the
          caller-owned persistent buffer and reuse it without resetting.
        - Provided ``rng_states`` with ``initialize_rng=True``: validate the
          caller-owned persistent buffer, then reset it from ``rng_seed``.
        - Validation failure: raise before RNG initialization or other Warp
          launches mutate state.

        Reusing the same ``rng_seed`` alongside a persistent ``rng_states``
        buffer does not reseed that buffer on later calls unless
        ``initialize_rng=True`` is passed. For repeated timesteps or graph
        capture setup, initialize caller-owned buffers once before the loop or
        before capture, then reuse them without hidden per-step resets.
    """
    n_boxes, n_particles, n_species = particles.masses.shape
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)

    device = particles.masses.device
    _validate_device_arrays(particles, device)
    time_step_value = _validate_time_step(time_step)
    max_collisions_value = _validate_max_collisions(max_collisions)
    temperature_array, pressure_array = _ensure_environment_arrays(
        temperature=temperature,
        pressure=pressure,
        environment=environment,
        n_boxes=n_boxes,
        device=device,
        caller_name="coagulation_step_gpu",
    )

    if volume is None:
        volume = particles.volume
    volume_array = _ensure_volume_array(volume, n_boxes, device)

    expected_pairs_shape = (n_boxes, max_collisions_value, 2)
    if collision_pairs is None:
        collision_pairs = wp.zeros(
            expected_pairs_shape,
            dtype=wp.int32,
            device=device,
        )
    else:
        _validate_collision_pairs(collision_pairs, expected_pairs_shape, device)

    expected_counts_shape = (n_boxes,)
    if n_collisions is None:
        n_collisions = wp.zeros(
            expected_counts_shape,
            dtype=wp.int32,
            device=device,
        )
    else:
        _validate_collision_counts(n_collisions, expected_counts_shape, device)

    initialize_rng_states_for_call = rng_states is None
    if rng_states is None:
        rng_states = wp.zeros(
            expected_counts_shape,
            dtype=wp.uint32,
            device=device,
        )
    else:
        _validate_rng_states(rng_states, expected_counts_shape, device)
        initialize_rng_states_for_call = initialize_rng

    radii = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    diffusivities = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    g_terms = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    speeds = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    active_flags = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )

    if initialize_rng_states_for_call:
        wp.launch(
            _initialize_rng_states,
            dim=n_boxes,
            inputs=[wp.uint32(rng_seed), rng_states],
            device=device,
        )

    wp.launch(
        brownian_coagulation_kernel,
        dim=(n_boxes,),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            volume_array,
            temperature_array,
            pressure_array,
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
            wp.float64(time_step_value),
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

    wp.launch(
        apply_coagulation_kernel,
        dim=(n_boxes, max_collisions_value),
        inputs=[
            particles.masses,
            particles.concentration,
            collision_pairs,
            n_collisions,
        ],
        device=device,
    )

    return particles, collision_pairs, n_collisions
