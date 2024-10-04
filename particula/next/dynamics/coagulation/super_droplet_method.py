"""
Super droplet method for coagulation dynamics.

Need to validate the code.
"""

from itertools import combinations_with_replacement
from typing import Tuple, Union, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline  # type: ignore


def super_droplet_update_step(
    particle_radius: NDArray[np.float64],
    concentration: NDArray[np.float64],
    single_event_counter: NDArray[np.int64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """
    Update the particle radii and concentrations after coagulation events.

    Args:
        particle_radius (NDArray[float64]): Array of particle radii.
        concentration (NDArray[float64]): Array representing the concentration
            of particles.
        single_event_counter (NDArray[int64]): Tracks the number of
            coagulation events for each particle.
        small_index (NDArray[int64]): Indices corresponding to smaller
            particles.
        large_index (NDArray[int64]): Indices corresponding to larger
            particles.

    Returns:
        - Updated array of particle radii.
        - Updated array representing the concentration of particles.
        - Updated array tracking the number of coagulation events.
    """

    # Step 1: Calculate the summed volumes of the smaller and larger particles
    # The volumes are obtained by cubing the radii of the particles.
    sum_radii_cubed = np.power(
        particle_radius[small_index], 3, dtype=np.float64
    ) + np.power(particle_radius[large_index], 3, dtype=np.float64)

    # Step 2: Calculate the new radii formed by the coagulation events
    # The new radius is the cube root of the summed volumes.
    new_radii = np.cbrt(sum_radii_cubed)

    # Step 3: Determine the concentration differences between small and
    # large particles
    concentration_delta = (
        concentration[small_index] - concentration[large_index]
    )
    more_small = concentration_delta > 0
    more_large = concentration_delta < 0
    equal_concentration = concentration_delta == 0

    # Step 4: Handle cases where small and large particle concentrations are
    # equal. In these cases, split the concentrations equally and update
    # both small and large particle radii.
    if np.any(equal_concentration):
        # print("Handling equal concentration case (split)")
        concentration[small_index[equal_concentration]] /= 2
        concentration[large_index[equal_concentration]] /= 2

        particle_radius[small_index[equal_concentration]] = new_radii[
            equal_concentration
        ]
        particle_radius[large_index[equal_concentration]] = new_radii[
            equal_concentration
        ]

    # Step 5: Handle cases where there are more large particles than small ones
    # Update the concentration of large particles and adjust the radii of
    # small particles.
    if np.any(more_large):
        # print("Handling more large particles case")
        concentration[large_index[more_large]] = np.abs(
            concentration_delta[more_large]
        )
        particle_radius[small_index[more_large]] = new_radii[more_large]

    # Step 6: Handle cases where there are more small particles than large ones
    # Update the concentration of small particles and adjust the radii of
    # large particles.
    if np.any(more_small):
        # print("Handling more small particles case")
        concentration[small_index[more_small]] = np.abs(
            concentration_delta[more_small]
        )
        particle_radius[large_index[more_small]] = new_radii[more_small]

    # Increment event counters for both small and large particles
    single_event_counter[small_index] += 1
    single_event_counter[large_index] += 1

    return particle_radius, concentration, single_event_counter


def event_pairs(
    lower_bin: int,
    upper_bin: int,
    kernel_max: Union[float, NDArray[np.float64]],
    number_in_bins: Union[NDArray[np.float64], NDArray[np.int64]],
) -> float:
    """Calculate the number of particle pairs based on kernel value.

    Args:
        lower_bin: Lower bin index.
        upper_bin: Upper bin index.
        kernel_max: Maximum value of the kernel.
        number_in_bins: Number of particles in each bin.

    Returns:
        The number of particle pairs events based on the kernel and
        number of particles in the bins.
    """
    # Calculate the number of particle pairs based on the kernel value
    if lower_bin != upper_bin:
        return (
            kernel_max * number_in_bins[lower_bin] * number_in_bins[upper_bin]
        )
    return (
        kernel_max
        # * 0.5
        * number_in_bins[lower_bin]
        * (number_in_bins[upper_bin] - 1)
    )


def sample_events(
    events: float,
    volume: float,
    time_step: float,
    generator: np.random.Generator,
) -> int:
    """
    Sample the number of coagulation events from a Poisson distribution.

    This function calculates the expected number of coagulation events based on
    the number of particle pairs, the simulation volume, and the time step. It
    then samples the actual number of events using a Poisson distribution.

    Args:
        events: The calculated number of particle pairs that could
            interact.
        volume: The volume of the simulation space.
        time_step: The time step over which the events are being simulated.
        generator: A NumPy random generator used to sample from the Poisson
            distribution.

    Returns:
        The sampled number of coagulation events as an integer.
    """
    # Calculate the expected number of events
    events_exact = events / volume

    # Sample the actual number of events from a Poisson distribution
    return generator.poisson(events_exact * time_step)


# pylint: disable=too-many-positional-arguments, too-many-arguments
def random_choice_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    particle_radius: NDArray[np.float64],
    bin_indices: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Filter valid indices and select random indices for coagulation events.

    This function filters particle indices based on bin indices and ensures
    the selected particles have a positive radius. It then randomly selects
    indices from both a lower bin and an upper bin for a given number of
    events.

    Args:
        lower_bin: The index of the lower bin to filter particles from.
        upper_bin: The index of the upper bin to filter particles from.
        events: Number of events (indices) to sample for each bin.
        particle_radius: A NumPy array of particle radii. Only particles with
            radius > 0 are considered.
        bin_indices: A NumPy array of bin indices corresponding to each
            particle.
        generator: A NumPy random generator used to sample indices.

    Returns:
        Tuple:
            - Indices of particles from the lower bin.
            - Indices of particles from the upper bin.

    Example:
        ``` py title="Example choice indices (update)"
        rng = np.random.default_rng()
        particle_radius = np.array([0.5, 0.0, 1.2, 0.3, 0.9])
        bin_indices = np.array([1, 1, 1, 2, 2])
        lower_bin = 1
        upper_bin = 2
        events = 2
        lower_indices, upper_indices = random_choice_indices(
            lower_bin, upper_bin, events, particle_radius, bin_indices, rng)
        # lower_indices: array([0, 4])
        # upper_indices: array([0, 1])
        ```
    """
    try:
        # Directly find the indices where the condition is True
        lower_indices = generator.choice(
            np.flatnonzero((bin_indices == lower_bin) & (particle_radius > 0)),
            events,
            replace=True,
        )
        upper_indices = generator.choice(
            np.flatnonzero((bin_indices == upper_bin) & (particle_radius > 0)),
            events,
            replace=True,
        )
    except ValueError:
        # If no valid indices are found, return empty arrays
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return lower_indices, upper_indices


def select_random_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    number_in_bins: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Select random indices for particles involved in coagulation events.

    This function generates random indices for particles in the specified bins
    (`lower_bin` and `upper_bin`) that are involved in a specified number of
    events. The indices are selected based on the number of particles in
    each bin.

    Args:
        lower_bin: Index of the bin containing smaller particles.
        upper_bin: Index of the bin containing larger particles.
        events: The number of events to sample indices for.
        number_in_bins: Array representing the number of particles in
            each bin.
        generator: A NumPy random generator used to sample indices.

    Returns:
        Tuple:
            - Indices of particles from `lower_bin`.
            - Indices of particles from `upper_bin`.
    """
    # Select random indices for particles in the lower_bin
    lower_indices = generator.integers(  # type: ignore
        0,
        number_in_bins[lower_bin],
        size=events,
        endpoint=False,
        dtype=np.int64,
    )
    # Select random indices for particles in the upper_bin
    upper_indices = generator.integers(  # type: ignore
        0,
        number_in_bins[upper_bin],
        size=events,
        endpoint=False,
        dtype=np.int64,
    )
    return lower_indices, upper_indices


def bin_to_particle_indices(
    lower_indices: NDArray[np.int64],
    upper_indices: NDArray[np.int64],
    lower_bin: int,
    upper_bin: int,
    bin_indices: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Convert bin indices to actual particle indices in the particle array.

    This function calculates the actual indices in the particle array
    corresponding to the bins specified by `lower_bin` and `upper_bin`.
    The function adjusts the provided bin-relative indices to reflect
    their position in the full particle array.

    Args:
        lower_indices: Array of indices relative to the start of
            the `lower_bin`.
        upper_indices: Array of indices relative to the start of
            the `upper_bin`.
        lower_bin: Index of the bin containing smaller particles.
        upper_bin: Index of the bin containing larger particles.
        bin_indices: Array containing the start indices of each bin in the
            particle array.

    Returns:
        Tuple:
            - `small_index`: Indices of particles from the `lower_bin`.
            - `large_index`: Indices of particles from the `upper_bin`.
    """
    # Get the start index in the particle array for the lower_bin
    start_index_lower_bin = np.searchsorted(bin_indices, lower_bin)
    # Get the start index in the particle array for the upper_bin
    start_index_upper_bin = np.searchsorted(bin_indices, upper_bin)

    # Calculate the actual particle indices for the lower_bin and upper_bin
    small_index = start_index_lower_bin + lower_indices
    large_index = start_index_upper_bin + upper_indices

    return small_index, large_index


def filter_valid_indices(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
    single_event_counter: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Filter particles indices based on particle radius and event counters.

    This function filters out particle indices that are considered invalid
    based on two criteria:
    1. The particle radius must be greater than zero.
    2. If provided, the single event counter must be less than one.

    Args:
        small_index: Array of indices for particles in the smaller bin.
        large_index: Array of indices for particles in the larger bin.
        particle_radius: Array containing the radii of particles.
        single_event_counter (Optional): Optional array tracking the
            number of events for each particle. If provided, only particles
            with a counter value less than one are valid.

    Returns:
        Tuple:
            - Filtered `small_index` array containing only valid indices.
            - Filtered `large_index` array containing only valid indices.
    """
    if single_event_counter is not None:
        # Both particle radius and event counter are used to determine
        # valid indices
        valid_indices = (
            (particle_radius[small_index] > 0)
            & (particle_radius[large_index] > 0)
            & (single_event_counter[small_index] < 1)
            & (single_event_counter[large_index] < 1)
        )
    else:
        # Only particle radius is used to determine valid indices
        valid_indices = (particle_radius[small_index] > 0) & (
            particle_radius[large_index] > 0
        )

    # Return the filtered indices
    return small_index[valid_indices], large_index[valid_indices]


def coagulation_events(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    kernel_values: NDArray[np.float64],
    kernel_max: float,
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Calculate coagulation probabilities and filter events based on them.

    This function calculates the probability of coagulation events occurring
    between pairs of particles, based on the ratio of the kernel value for
    each pair to the maximum kernel value for the bins. The function then
    randomly determines which events occur using these probabilities.

    Args:
        small_index: Array of indices for the first set of particles
            (smaller particles) involved in the events.
        large_index: Array of indices for the second set of particles
            (larger particles) involved in the events.
        kernel_values: Array of kernel values corresponding to the
            particle pairs.
        kernel_max: The maximum kernel value used for normalization
            of probabilities.
        generator: A NumPy random generator used to sample random numbers.

    Returns:
        Tuple:
            - Filtered `small_index` array containing indices where
                coagulation events occurred.
            - Filtered `large_index` array containing indices where
                coagulation events occurred.
    """
    # Calculate the coagulation probabilities for each particle pair
    coagulation_probabilities = kernel_values / kernel_max
    # coagulation_probabilities = kernel_values * kernel_max

    # Determine which events occur based on these probabilities
    coagulation_occurs = (
        generator.uniform(low=0, high=1, size=len(coagulation_probabilities))
        < coagulation_probabilities
    )
    # coagulation_occurs = coagulation_probabilities>0

    # Return the indices of particles that underwent coagulation
    return small_index[coagulation_occurs], large_index[coagulation_occurs]


def sort_particles(
    particle_radius: NDArray[np.float64],
    particle_concentration: Optional[NDArray[np.float64]] = None,
) -> Tuple[
    NDArray[np.int64], NDArray[np.float64], Optional[NDArray[np.float64]]
]:
    """
    Sort particles by size and optionally sort their concentrations.

    Args:
        particle_radius: Array of particle radii.
        particle_concentration: Optional array of particle concentrations
            corresponding to each radius. If provided, it will be sorted to
            match the sorted radii.

    Returns:
        Tuple:
            - `unsort_indices`: Array of indices to revert the sorting.
            - `sorted_radius`: Array of sorted particle radii.
            - `sorted_concentration`: Optional array of sorted particle
                concentrations (or `None` if not provided).
    """
    # Sort the particle radii and get the sorted indices
    sorted_indices = np.argsort(particle_radius)
    # Calculate the indices needed to revert the sorting
    unsort_indices = np.argsort(sorted_indices)
    # Sort the particle radii
    sorted_radius = particle_radius[sorted_indices]

    # If concentrations are provided, sort them too; otherwise, return None
    sorted_concentration = (
        particle_concentration[sorted_indices]
        if particle_concentration is not None
        else None
    )
    return unsort_indices, sorted_radius, sorted_concentration


def bin_particles(
    particle_radius: NDArray[np.float64],
    radius_bins: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Bin particles by size and return the number of particles in each bin.

    Args:
        particle_radius: Array of sorted particle radii.
        radius_bins: Array defining the bin edges for particle radii.

    Returns:
        Tuple:
            - Array of the number of particles in each bin.
            - Array of bin indices for each particle.
    """
    number_in_bins, bins = np.histogram(particle_radius, bins=radius_bins)
    bin_indices = np.digitize(particle_radius, bins, right=True)

    # Handle overflow and underflow bin indices
    bin_indices[bin_indices == len(bins)] = len(bins) - 1
    bin_indices[bin_indices == 0] = 1
    bin_indices -= 1  # Adjust to 0-based indexing

    return number_in_bins, bin_indices


def get_bin_pairs(
    bin_indices: NDArray[np.int64],
) -> list[Tuple[int, int]]:
    """
    Pre-compute the unique bin pairs for vectorized operations.

    Args:
        bin_indices: Array of bin indices.

    Returns:
        Unique bin pairs for vectorized operations.
    """
    unique_bins = np.unique(bin_indices)
    return list(combinations_with_replacement(unique_bins, 2))


def calculate_concentration_in_bins(
    bin_indices: NDArray[np.int64],
    particle_concentration: NDArray[np.float64],
    number_in_bins: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate the concentration of particles in each bin.

    Args:
        bin_indices: Array of bin indices for each particle.
        particle_concentration: Array of sorted particle concentrations.
        number_in_bins : Array of the number of particles in each bin.

    Returns:
        The total concentration in each bin.
    """
    concentration_in_bins = np.zeros_like(number_in_bins, dtype=np.float64)
    unique_bins = np.unique(bin_indices)

    for unique_bin in unique_bins:
        concentration_in_bins[unique_bin] = np.sum(
            particle_concentration[bin_indices == unique_bin]
        )

    return concentration_in_bins


# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals
def super_droplet_coagulation_step(
    particle_radius: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Perform a single step of the Super Droplet coagulation process.

    This function processes particles by sorting them, binning by size,
    computing coagulation events based on the coagulation kernel, and
    updating particle properties accordingly.

    Args:
        particle_radius: Array of particle radii.
        particle_concentration: Array of particle concentrations
            corresponding to each radius.
        kernel: 2D array representing the coagulation kernel values between
            different bins.
        kernel_radius: Array defining the radii corresponding to the
            kernel bins.
        volume: Volume of the system or relevant scaling factor.
        time_step: Duration of the current time step.
        random_generator : A NumPy random number generator for
            stochastic processes.

    Returns:
        Tuple:
            - Updated array of particle radii after coagulation.
            - Updated array of particle concentrations after coagulation.
    """
    # Step 1: Sort particles by size and obtain indices to revert sorting later
    unsort_indices, sorted_radius, sorted_concentration = sort_particles(
        particle_radius=particle_radius,
        particle_concentration=particle_concentration,
    )
    # Step 2: Bin particles by size using the provided kernel radius bins
    number_in_bins, bin_indices = bin_particles(
        particle_radius=sorted_radius, radius_bins=kernel_radius
    )
    # Step 3: Precompute unique bin pairs for efficient coagulation
    # computations
    pair_indices = get_bin_pairs(bin_indices=bin_indices)

    # Step 4: Calculate the total concentration of particles within each bin
    concentration_in_bins = calculate_concentration_in_bins(
        bin_indices=bin_indices,
        particle_concentration=sorted_concentration,  # type: ignore
        number_in_bins=number_in_bins,
    )

    # Step 5: Initialize a bivariate spline for interpolating kernel values
    # between bin radii
    interp_kernel = RectBivariateSpline(
        x=kernel_radius, y=kernel_radius, z=kernel
    )

    # Step 6: Initialize a counter to track the number of coagulation events
    # per particle
    single_event_counter = np.zeros_like(particle_radius, dtype=int)

    # Step 7: Iterate over each unique pair of bins to perform
    # coagulation events
    for lower_bin, upper_bin in pair_indices:
        # Step 7.1: Retrieve the maximum kernel value for the current bin pair
        # Note: The '+1' indexing assumes that 'kernel' has dimensions
        # accommodating this offset due to bin edges
        kernel_max = kernel[lower_bin, upper_bin + 1]

        # Step 7.2: Determine potential coagulation events between
        # particles in these bins
        events = event_pairs(
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            kernel_max=kernel_max,
            number_in_bins=concentration_in_bins,
        )

        # Step 7.3: Sample the number of coagulation events from a
        # Poisson distribution
        num_events = sample_events(
            events=events,
            volume=volume,
            time_step=time_step,
            generator=random_generator,
        )

        # Step 7.4: If no events are expected, skip to the next bin pair
        if num_events == 0:
            continue

        # Step 7.5: Limit the number of events to the available number of
        # particles in each bin. This prevents oversampling beyond the
        # available particles
        num_events = min(
            num_events,
            number_in_bins[lower_bin],
            number_in_bins[upper_bin],
        )

        # Step 7.6: Randomly select indices of particles involved in the
        # coagulation events within the current bins
        r_i_indices, r_j_indices = select_random_indices(
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            events=num_events,
            number_in_bins=number_in_bins,
            generator=random_generator,
        )

        # Step 7.7: Convert bin-relative indices to actual particle indices
        # in the sorted arrays
        indices_i, indices_j = bin_to_particle_indices(
            lower_indices=r_i_indices,
            upper_indices=r_j_indices,
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            bin_indices=bin_indices,
        )

        # Step 7.8: Filter out invalid particle pairs based on their radii
        # and event counters
        indices_i, indices_j = filter_valid_indices(
            small_index=indices_i,
            large_index=indices_j,
            particle_radius=particle_radius,
            single_event_counter=single_event_counter,
        )

        # Step 7.9: If no valid indices remain after filtering, skip to
        # the next bin pair
        if indices_i.size == 0:
            continue

        # Step 7.10: Interpolate kernel values for the selected particle pairs
        kernel_values = interp_kernel.ev(
            particle_radius[indices_i], particle_radius[indices_j]
        )

        # Step 7.11: Determine which coagulation events actually occur based
        # on interpolated kernel probabilities
        indices_i, indices_j = coagulation_events(
            small_index=indices_i,
            large_index=indices_j,
            kernel_values=kernel_values,
            kernel_max=kernel_max,
            generator=random_generator,
        )

        # Step 7.12: Update particle properties based on the coagulation events
        # This step typically involves merging particles, updating
        # concentrations, and tracking events
        particle_radius, particle_concentration, single_event_counter = (
            super_droplet_update_step(
                particle_radius=particle_radius,
                concentration=particle_concentration,
                single_event_counter=single_event_counter,
                small_index=indices_i,
                large_index=indices_j,
            )
        )

    # Step 8: Unsort the particles to restore their original ordering
    # before sorting
    particle_radius = particle_radius[unsort_indices]
    particle_concentration = particle_concentration[unsort_indices]

    # Step 9: Return the updated particle radii and concentrations after
    # the coagulation step
    return particle_radius, particle_concentration
