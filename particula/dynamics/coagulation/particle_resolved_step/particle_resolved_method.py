"""Particle resolved method for coagulation."""

from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator  # type: ignore

from .super_droplet_method import (
    _bin_particles,
    _get_bin_pairs,
)


def get_particle_resolved_update_step(
    particle_radius: NDArray[np.float64],
    loss: NDArray[np.float64],
    gain: NDArray[np.float64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Update particle radii and track lost/gained particles after coagulation
    events.

    This function simulates the immediate effect of coagulation on particle
    radii, marking smaller particles as lost and updating the larger particles
    to the new radius computed from volume conservation. The calculation is:

    - r_new = cbrt(r_small³ + r_large³)
        - r_new is the new radius in meters,
        - r_small is the smaller particle's radius in meters,
        - r_large is the larger particle's radius in meters.

    Arguments:
        - particle_radius : Array of particle radii.
        - loss : Array to store lost particle radii.
        - gain : Array to store gained particle radii.
        - small_index : Indices of smaller particles.
        - large_index : Indices of larger particles.

    Returns:
        - Updated array of particle radii after coagulation events.
        - Updated array for the radii of particles that were lost.
        - Updated array for the radii of particles that were gained.

    Examples:
        ```py title="Example Usage"
        import numpy as np
        from particula.dynamics.coagulation.particle_resolved_step import
            particle_resolved_method

        r = np.array([1e-9, 2e-9, 3e-9, 1e-9])
        lost = np.zeros_like(r)
        gained = np.zeros_like(r)
        s_idx = np.array([0, 1])
        l_idx = np.array([2, 3])
        updated_r, lost_r, gained_r = (
            particle_resolved_method.get_particle_resolved_update_step(
                r, lost, gained, s_idx, l_idx
            ))
        # updated_r now has coagulated radii, lost_r and gained_r are tracked.
    """
    # Step 1: Calculate the summed volumes of the smaller and larger particles
    # The volumes are obtained by cubing the radii of the particles.
    sum_radii_cubed = np.power(
        particle_radius[small_index], 3, dtype=np.float64
    ) + np.power(particle_radius[large_index], 3, dtype=np.float64)

    # Step 2: Calculate the new radii formed by the coagulation events
    # The new radius is the cube root of the summed volumes.
    new_radii = np.cbrt(sum_radii_cubed)

    # Step 3: Save out the loss and gain of particles
    loss[small_index] = particle_radius[small_index]
    gain[large_index] = particle_radius[large_index]

    # Step 4: Remove the small particles as they coagulated to the larger ones
    particle_radius[small_index] = 0

    # Step 5: Increase the radii of the large particles to the new radii
    particle_radius[large_index] = new_radii

    return particle_radius, loss, gain


# pylint: disable=too-many-positional-arguments, too-many-arguments
# pylint: disable=too-many-locals
def get_particle_resolved_coagulation_step(
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> NDArray[np.int64]:
    """Perform a single step of particle coagulation, updating particle radii
    with a stochastic approach.

    This function models collisions between particles based on a given
    coagulation kernel. It identifies potential collision pairs, randomly
    selects which collisions occur according to a probability derived from the
    kernel value, and then tracks which particles have coagulated.

    The main calculation for the probability of coagulation is:

    - Probability = K × Δt × (possible collisions) / (tests × volume)
        - K is the interpolated kernel value,
        - Δt is the timestep,
        - volume is the system volume.

    Arguments:
        - particle_radius : Array of particle radii.
        - kernel : 2D coagulation kernel matrix matching the size of
            kernel_radius.
        - kernel_radius : Radii used to index or interpolate the kernel.
        - volume : Volume of the system in m³.
        - time_step : Time step for each coagulation iteration in seconds.
        - random_generator : Random number generator for the stochastic
            approach.

    Returns:
        - An array of shape (N, 2), where each row contains
            [small_index, large_index] for coagulation events.

    Examples:
        ```py title="Example Usage"
        import numpy as np
        from particula.dynamics.coagulation.particle_resolved_step import
            particle_resolved_method

        r = np.array([1e-9, 2e-9, 3e-9])
        kernel_values = np.ones((50, 50))
        kernel_r = np.linspace(1e-10, 1e-7, 50)
        vol = 1e-3
        dt = 0.01
        rng = np.random.default_rng(42)
        event_pairs =
        particle_resolved_method.get_particle_resolved_coagulation_step(
            particle_radius=r,
            kernel=kernel_values,
            kernel_radius=kernel_r,
            volume=vol,
            time_step=dt,
            random_generator=rng
        )
        # event_pairs contains the pairs of [small, large] indices that
        # coagulated.

    References:
        - Seinfeld, J. H., & Pandis, S. N. *Atmospheric Chemistry and Physics*,
          Wiley, 2016.
    """
    # Step 1: Bin the particles based on their radii into corresponding kernel
    # bins
    _, bin_indices = _bin_particles(particle_radius, kernel_radius)

    # Step 2: Precompute unique bin pairs for efficient coagulation
    pair_indices = _get_bin_pairs(bin_indices=bin_indices)

    # Step 3: Interpolate the coagulation kernel for efficient lookups during
    # the coagulation process
    interp_kernel = _interpolate_kernel(kernel, kernel_radius)

    # Create output arrays to store the indices of small and large particles
    # involved in coagulation events
    small_index_total = np.array([], dtype=np.int64)
    large_index_total = np.array([], dtype=np.int64)

    # Step 4: Iterate over each bin pair to calculate potential coagulation
    # events
    for lower_bin, upper_bin in pair_indices:
        # Get indices of particles in the current bin and filter out any that
        # have already coagulated
        small_indices = np.flatnonzero(
            (bin_indices == lower_bin) & (particle_radius > 0)
        )
        small_indices = np.setdiff1d(small_indices, small_index_total)

        large_indices = np.flatnonzero(
            (bin_indices == upper_bin) & (particle_radius > 0)
        )

        # Skip to the next bin pair if there are no particles in one or
        # both bins
        if len(small_indices) == 0 or len(large_indices) == 0:
            continue

        # Step 5: Retrieve the maximum kernel value for the current bin pair
        small_sample = np.min(particle_radius[small_indices])
        large_sample = np.max(particle_radius[large_indices])
        kernel_values = interp_kernel(np.array([[small_sample, large_sample]]))

        # Step 6: Calculate the number of possible coagulation events
        # between small and large particles
        events_count: float = float(len(small_indices) * len(large_indices))
        if lower_bin == upper_bin:
            events_count = len(small_indices) * (len(large_indices) - 1) / 2
        events = int(np.ceil(events_count))

        # Step 7: Determine the number of coagulation tests to run based
        # on kernel value and system parameters
        tests = int(np.ceil(kernel_values.item() * time_step * events / volume))
        if tests == 0 or events == 0:
            continue

        # Step 8: Randomly select small and large particle pairs for
        # coagulation tests
        replace_in_pool = tests > len(small_indices)
        small_index = random_generator.choice(  # type: ignore
            small_indices, size=tests, replace=bool(replace_in_pool)
        )
        large_index = random_generator.choice(large_indices, tests)

        # Step 9: Calculate the kernel value for the selected particle pairs
        kernel_value = interp_kernel(
            np.column_stack(
                (particle_radius[small_index], particle_radius[large_index])
            )
        )

        # Handle diagonal elements if necessary (for single pair coagulation)
        if kernel_value.ndim > 1:
            kernel_value = np.diagonal(kernel_value)

        # Step 10: Calculate coagulation probabilities for each selected pair
        coagulation_probabilities = _calculate_probabilities(
            kernel_value, time_step, events, tests, volume
        )

        # Step 11: Determine which coagulation events occur based on
        # random uniform sampling
        valid_indices = np.flatnonzero(  # type: ignore
            random_generator.uniform(size=tests) < coagulation_probabilities
        )

        # Step 12: Ensure each small particle only coagulates with one large
        # particle
        _, unique_index = np.unique(
            small_index[valid_indices], return_index=True
        )
        # Valid and unique indices are selected for coagulation,
        # non-unique indices should happen rarely.
        small_index = small_index[valid_indices][unique_index]
        large_index = large_index[valid_indices][unique_index]

        # Step 13: Save the coagulation events
        small_index_total = np.append(small_index_total, small_index)
        large_index_total = np.append(large_index_total, large_index)

    # Step 14: Resolve any series of coagulation events that involve the same
    # particles multiple times
    small_index_total, large_index_total = _final_coagulation_state(
        small_index_total, large_index_total, particle_radius
    )

    # Step 15: Combine small and large indices into pairs representing the
    # loss and gain events
    loss_gain_index = np.column_stack([small_index_total, large_index_total])

    return loss_gain_index


def _interpolate_kernel(
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
) -> RegularGridInterpolator:
    """Create an interpolation function for the coagulation kernel with
    out-of-bounds handling.

    This function returns a RegularGridInterpolator that performs linear
    interpolation for values within the domain of the kernel and clamps to the
    nearest value outside of it.

    Arguments:
        - kernel : 2D coagulation kernel values.
        - kernel_radius : Radii corresponding to kernel bins.

    Returns:
        - A RegularGridInterpolator object for retrieving kernel values based
            on radius pairs.

    Examples:
        ```py
        import numpy as np
        from particula.dynamics.coagulation.particle_resolved_step import
            particle_resolved_method
        kernel_vals = np.random.rand(10,10)
        rad = np.linspace(1e-9, 1e-7, 10)
        interpolator = particle_resolved_method._interpolate_kernel(
            kernel_vals, rad
        )
        # Use interpolator([[r_small, r_large]]) to get kernel value
        ```
    """
    grid = (kernel_radius, kernel_radius)
    # Using linear interpolation inside domain; clamp to boundary
    # out-of-bound points.
    return RegularGridInterpolator(
        points=grid,
        values=kernel,
        method="linear",
        bounds_error=False,
        fill_value=None,  # type: ignore  # None uses nearest neighbor
    )


def _calculate_probabilities(
    kernel_values: Union[float, NDArray[np.float64]],
    time_step: float,
    events: int,
    tests: int,
    volume: float,
) -> Union[float, NDArray[np.float64]]:
    """Calculate coagulation probabilities based on kernel values and system
    parameters.

    This function multiplies the kernel values by the time step and a factor
    derived from the ratio of (events / tests) over the volume to obtain the
    probability of coagulation.

    Arguments:
        - kernel_values : Interpolated kernel values for a given particle pair,
          may be scalar or array.
        - time_step : Duration of one coagulation step in seconds.
        - events : Number of possible collisions for the pair(s).
        - tests : Number of trials for the random selection procedure.
        - volume : System volume in m³.

    Returns:
        - The probability (or array of probabilities) that a collision occurs
          during this time step.

    Examples:
        ```py
        prob = _calculate_probabilities(0.5, 1.0, 20, 10, 1e-3)
        # prob ~ 0.5 * 1.0 * 20 / (10 * 1e-3) = 1000
        ```
    """
    return kernel_values * time_step * events / (tests * volume)


def _final_coagulation_state(
    small_indices: NDArray[np.int64],
    large_indices: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Resolve the final state of particles that have undergone multiple
    coagulation events.

    This function ensures that each small particle index merges correctly to
    a final large particle index, preventing logical conflicts (e.g., a single
    particle merging into multiple large particles in the same step).

    Arguments:
        - small_indices : Array of smaller particle indices in coagulation.
        - large_indices : Array of larger particle indices in coagulation.
        - particle_radius : Array of current particle radii.

    Returns:
        - A tuple (updated_small_indices, updated_large_indices) that resolves
          multiple merges for the same particle.

    Examples:
        ```py
        import numpy as np
        small = np.array([0, 1, 2])
        large = np.array([2, 3, 4])
        r = np.array([1e-9, 1.5e-9, 2e-9, 3e-9, 4e-9])
        s_final, l_final = _final_coagulation_state(small, large, r)
        # ensures each index in s_final merges to a single large index
        ```
    """
    # Find common indices that appear in both small and large arrays
    commons, small_common, large_common = np.intersect1d(
        small_indices, large_indices, return_indices=True
    )
    # Sort particles by radius to resolve final states in the correct order
    sorted_args = np.argsort(particle_radius[commons])
    commons = commons[sorted_args]
    small_common = small_common[sorted_args]
    large_common = large_common[sorted_args]

    # Remap to the largest particle in common to resolve the final state
    for i, common in enumerate(commons):
        final_value = large_indices[small_common[i]]
        remap_index = np.flatnonzero(large_indices == common)
        large_indices[remap_index] = final_value

    return small_indices, large_indices
