"""Particle-resolved coagulation helpers with bounded kernel interpolation."""

from typing import Callable, Optional, Tuple, Union

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
    """Update particle radii and track losses and gains from coagulation.

    Smaller particles indexed by ``small_index`` are removed and their radii
    recorded in ``loss``. The corresponding larger particles indexed by
    ``large_index`` grow to conserve volume, and their original radii are
    stored in ``gain``.

    Args:
        particle_radius: Radii of all particles. Modified in place.
        loss: Array that records radii of particles removed by coagulation.
        gain: Array that records radii of particles that absorbed others.
        small_index: Indices of smaller particles in each coagulation pair.
        large_index: Indices of larger particles in each coagulation pair.

    Returns:
        Updated particle radii, loss, and gain arrays after coagulation.
    """
    # Step 1: Calculate base volumes from pre-update radii.
    base_volume = np.power(particle_radius, 3, dtype=np.float64)
    small_volume = base_volume[small_index].copy()
    base_volume[small_index] = 0
    np.add.at(base_volume, large_index, small_volume)

    # Step 2: Save out the loss and gain of particles before modifying radii.
    loss[small_index] = particle_radius[small_index]
    gain[large_index] = particle_radius[large_index]

    # Step 3: Recompute all radii from accumulated volumes.
    particle_radius[:] = np.cbrt(base_volume)

    return particle_radius, loss, gain


# pylint: disable=too-many-positional-arguments, too-many-arguments
# pylint: disable=too-many-locals
def get_particle_resolved_coagulation_step(  # noqa: C901
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
    kernel_func: Optional[
        Callable[
            [NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
        ]
    ] = None,
) -> NDArray[np.int64]:
    """Perform one stochastic coagulation step for a particle population.

    Particles are binned by radius, kernel values are interpolated for each
    bin pair, and random trials determine which collisions occur. Returned
    indices describe which small particles merged into which large particles.

    Args:
        particle_radius: Radii of particles that may coagulate.
        kernel: Two-dimensional coagulation kernel aligned to ``kernel_radius``.
        kernel_radius: Radii that define kernel grid points.
        volume: System volume in cubic meters.
        time_step: Duration of the coagulation step in seconds.
        random_generator: NumPy random generator used for sampling.
        kernel_func: Optional callable for direct kernel evaluation. When
            provided, it is called with ``(r_small, r_large)`` inputs and
            should return a 1D array of kernel values.

    Returns:
        Array of shape ``(n, 2)`` with ``[small_index, large_index]`` pairs
        describing coagulation events.
    """
    # Step 1: Bin the particles based on their radii into corresponding kernel
    # bins
    _, bin_indices = _bin_particles(particle_radius, kernel_radius)

    # Step 2: Precompute unique bin pairs for efficient coagulation
    pair_indices = _get_bin_pairs(bin_indices=bin_indices)

    # Step 3: Interpolate the coagulation kernel for efficient lookups during
    # the coagulation process (unless a direct kernel callable is provided)
    interp_kernel: Optional[RegularGridInterpolator] = None
    if kernel_func is None:
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
        if kernel_func is None:
            if interp_kernel is None:
                raise ValueError("Kernel interpolator is not initialized.")
            kernel_values = interp_kernel(
                np.array([[small_sample, large_sample]])
            )
        else:
            kernel_values = np.atleast_1d(
                np.asarray(
                    kernel_func(
                        np.asarray([small_sample], dtype=np.float64),
                        np.asarray([large_sample], dtype=np.float64),
                    ),
                    dtype=np.float64,
                )
            )

        # Step 6: Calculate the number of possible coagulation events
        # between small and large particles
        events_count: float = float(len(small_indices) * len(large_indices))
        if lower_bin == upper_bin:
            events_count = len(small_indices) * (len(large_indices) - 1) / 2
        events = int(np.ceil(events_count))

        # Step 7: Determine the number of coagulation tests to run based
        # on kernel value and system parameters
        tests = int(np.ceil(kernel_values.item() * time_step * events / volume))
        if tests <= 0 or events == 0:
            continue
        # Cap tests to prevent memory issues when particles are outside
        # the kernel radius range (which can cause inflated kernel values)
        max_tests = max(len(small_indices), len(large_indices)) * 10
        tests = min(tests, max_tests)

        # Step 8: Randomly select small and large particle pairs for
        # coagulation tests
        replace_in_pool = tests > len(small_indices)
        small_index = random_generator.choice(  # type: ignore
            small_indices, size=tests, replace=bool(replace_in_pool)
        )
        large_index = random_generator.choice(large_indices, tests)

        # Step 9: Calculate the kernel value for the selected particle pairs
        if kernel_func is None:
            if interp_kernel is None:
                raise ValueError("Kernel interpolator is not initialized.")
            kernel_value = interp_kernel(
                np.column_stack(
                    (
                        particle_radius[small_index],
                        particle_radius[large_index],
                    )
                )
            )
        else:
            kernel_value = np.atleast_1d(
                np.asarray(
                    kernel_func(
                        particle_radius[small_index],
                        particle_radius[large_index],
                    ),
                    dtype=np.float64,
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
    """Create a kernel interpolator that zeros out-of-bounds radii.

    Non-finite kernel entries are replaced with zeros, linear interpolation is
    used within the grid, and a zero ``fill_value`` prevents charge-blind
    extrapolation outside the tabulated radius range.

    Args:
        kernel: Two-dimensional coagulation kernel values.
        kernel_radius: Radii that define the kernel grid.

    Returns:
        RegularGridInterpolator that maps ``[r_small, r_large]`` pairs to
        kernel values with safe bounds handling.
    """
    grid = (kernel_radius, kernel_radius)
    cleaned_kernel = np.where(np.isfinite(kernel), kernel, 0.0)
    # Using linear interpolation inside domain; return zero for out-of-bound
    # points to avoid extrapolating charge-dependent values across bins.
    return RegularGridInterpolator(
        points=grid,
        values=cleaned_kernel,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )


def _calculate_probabilities(
    kernel_values: Union[float, NDArray[np.float64]],
    time_step: float,
    events: int,
    tests: int,
    volume: float,
) -> Union[float, NDArray[np.float64]]:
    """Calculate coagulation probabilities for sampled particle pairs.

    Args:
        kernel_values: Interpolated kernel values for particle pairs.
        time_step: Duration of the coagulation step in seconds.
        events: Number of possible collisions for the pair set.
        tests: Number of collision trials drawn for this pair set.
        volume: System volume in cubic meters.

    Returns:
        Probability (or array of probabilities) that collisions occur during
        the step.
    """
    return kernel_values * time_step * events / (tests * volume)


def _final_coagulation_state(
    small_indices: NDArray[np.int64],
    large_indices: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Resolve chained coagulation events to consistent merge pairs.

    When a particle appears as both a small and large partner, this function
    remaps merges so each small particle maps to one final large particle.

    Args:
        small_indices: Indices of smaller particles in candidate merges.
        large_indices: Indices of larger particles in candidate merges.
        particle_radius: Current particle radii used to order remapping.

    Returns:
        Tuple of ``(small_indices, large_indices)`` with conflicts resolved.
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
