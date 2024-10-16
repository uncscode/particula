"""Particle resolved method for coagulation.
"""

from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline  # type: ignore

from particula.dynamics.coagulation.super_droplet_method import (
    bin_particles,
    get_bin_pairs,
)


def interpolate_kernel(
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
) -> RectBivariateSpline:
    """
    Create a 2D interpolation function for the coagulation kernel.

    Args:
        kernel (NDArray[np.float64]): Coagulation kernel.
        kernel_radius (NDArray[np.float64]): Radii corresponding to kernel
            bins.

    Returns:
        RectBivariateSpline: Interpolated kernel function.
    """
    return RectBivariateSpline(x=kernel_radius, y=kernel_radius, z=kernel)


def calculate_probabilities(
    kernel_values: Union[float, NDArray[np.float64]],
    time_step: float,
    events: int,
    tests: int,
    volume: float,
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate coagulation probabilities based on kernel values and system
    parameters.

    Args:
        kernel_values (float): Interpolated kernel value for a particle pair.
        time_step (float): The time step over which coagulation occurs.
        events (int): Number of possible coagulation events.
        tests (int): Number of tests (or trials) for coagulation.
        volume (float): Volume of the system.

    Returns:
        float: Coagulation probability.
    """
    return kernel_values * time_step * events / (tests * volume)


def resolve_final_coagulation_state(
    small_indices: NDArray[np.int64],
    large_indices: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Resolve the final state of particles that have undergone multiple
    coagulation events.

    Args:
        small_indices (NDArray[np.int64]): Indices of smaller particles.
        large_indices (NDArray[np.int64]): Indices of larger particles.
        particle_radius (NDArray[np.float64]): Radii of particles.

    Returns:
        Tuple[NDArray[np.int64], NDArray[np.int64]]: Updated small and large
        indices.
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


def particle_resolved_update_step(
    particle_radius: NDArray[np.float64],
    loss: NDArray[np.float64],
    gain: NDArray[np.float64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Update the particle radii and concentrations after coagulation events.

    Args:
        particle_radius (NDArray[float64]): Array of particle radii.
        small_index (NDArray[int64]): Indices corresponding to smaller
            particles.
        large_index (NDArray[int64]): Indices corresponding to larger
            particles.

    Returns:
        - Updated array of particle radii.
        - Updated array for the radii of particles that were lost.
        - Updated array for the radii of particles that were gained.
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


# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals
def particle_resolved_coagulation_step(
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> NDArray[np.int64]:
    """
    Perform a single step of particle coagulation, updating particle radii
    based on coagulation events.

    Args:
        particle_radius (NDArray[np.float64]): Array of particle radii.
        kernel (NDArray[np.float64]): Coagulation kernel as a 2D array where
            each element represents the probability of coagulation between
            particles of corresponding sizes.
        kernel_radius (NDArray[np.float64]): Array of radii corresponding to
            the kernel bins.
        volume (float): Volume of the system in which coagulation occurs.
        time_step (float): Time step over which coagulation is calculated.
        random_generator (np.random.Generator): Random number generator for
            stochastic processes.

    Returns:
        NDArray[np.int64]: Array of indices corresponding to the coagulation
            events, where each element is a pair of indices corresponding to
            the coagulating particles [loss, gain].
    """

    # Step 1: Bin the particles based on their radii into corresponding kernel
    # bins
    _, bin_indices = bin_particles(particle_radius, kernel_radius)

    # Step 2: Precompute unique bin pairs for efficient coagulation
    pair_indices = get_bin_pairs(bin_indices=bin_indices)

    # Step 3: Interpolate the coagulation kernel for efficient lookups during
    # the coagulation process
    interp_kernel = interpolate_kernel(kernel, kernel_radius)

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
        kernel_values = interp_kernel.ev(  # type: ignore
            np.min(particle_radius[small_indices]),
            np.max(particle_radius[large_indices]),
        )

        # Step 6: Calculate the number of possible coagulation events
        # between small and large particles
        events = len(small_indices) * len(large_indices)
        if lower_bin == upper_bin:
            events = len(small_indices) * (len(large_indices) - 1) / 2
        events = int(np.ceil(events))

        # Step 7: Determine the number of coagulation tests to run based
        # on kernel value and system parameters
        tests = int(np.ceil(kernel_values * time_step * events / volume))

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
        kernel_value = interp_kernel.ev(  # type: ignore
            particle_radius[small_index], particle_radius[large_index]
        )

        # Handle diagonal elements if necessary (for single pair coagulation)
        if kernel_value.ndim > 1:
            kernel_value = np.diagonal(kernel_value)

        # Step 10: Calculate coagulation probabilities for each selected pair
        coagulation_probabilities = calculate_probabilities(
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
    small_index_total, large_index_total = resolve_final_coagulation_state(
        small_index_total, large_index_total, particle_radius
    )

    # Step 15: Combine small and large indices into pairs representing the
    # loss and gain events
    loss_gain_index = np.column_stack([small_index_total, large_index_total])

    return loss_gain_index
