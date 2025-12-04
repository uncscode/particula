"""Super droplet method for coagulation dynamics.

This module implements a Super Droplet Method for coagulation
dynamics, used to simulate how particles grow through collisions.
"""

from itertools import combinations_with_replacement
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline  # type: ignore


def _super_droplet_update_step(
    particle_radius: NDArray[np.float64],
    concentration: NDArray[np.float64],
    single_event_counter: NDArray[np.int64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Update particle radii and concentrations when two particles coagulate.

    This function merges smaller and larger particles by combining their
    volumes and redistributing particle concentrations. The resulting
    radii are computed via volume conservation, and an event counter
    tracks how many coagulation events each particle has undergone.

    Arguments:
        - particle_radius : Array of particle radii (m).
        - concentration : Array representing the concentration of each
          particle (number or mass, depending on usage).
        - single_event_counter : Tracks the number of coagulation events
          each particle has experienced in the current iteration.
        - small_index : Indices for smaller particles in a coagulation event.
        - large_index : Indices for larger particles in a coagulation event.

    Returns:
        - An updated array of particle radii (m) following coagulation.
        - An updated array representing the concentration of particles.
        - An updated array tracking the index-wise number of events.

    Examples:
        ```py
        import numpy as np
        r = np.array([1e-9, 2e-9, 3e-9])
        conc = np.array([100., 50., 75.])
        events = np.zeros_like(r, dtype=int)
        s_idx = np.array([0])
        l_idx = np.array([2])
        out_r, out_c, out_ev = _super_droplet_update_step(
            r, conc, events, s_idx, l_idx)
        # out_r[0] is updated via volume combination with out_r[2].
        ```
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


def _event_pairs(
    lower_bin: int,
    upper_bin: int,
    kernel_max: Union[float, NDArray[np.float64]],
    number_in_bins: Union[NDArray[np.float64], NDArray[np.int64]],
) -> float:
    """Calculate an approximate count of particle-pair interactions.

    This function estimates the number of collisions or interactions
    that might occur between two bins of particles, given a maximum
    kernel value and the current population of each bin. When the bins
    are the same, a correction factor is applied to avoid double-counting
    pairs.

    Arguments:
        - lower_bin : Index of the lower bin in the distribution.
        - upper_bin : Index of the upper bin in the distribution.
        - kernel_max : Maximum kernel value used to weight collisions.
        - number_in_bins : The population of particles per bin.

    Returns:
        - A float representing the expected number of particle-pair
          collision events.

    Examples:
        ```py
        max_kernel = 1.0e-9
        n_bins = np.array([100, 150, 200])
        # lower_bin=0, upper_bin=1 => collisions between bin 0 and bin 1
        events_est = _event_pairs(0, 1, max_kernel, n_bins)
        # events_est is ~ 1.0e-9 * 100 * 150
        ```
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


def _sample_events(
    events: float,
    volume: float,
    time_step: float,
    generator: np.random.Generator,
) -> int:
    """Determine how many collisions actually occur using a Poisson draw.

    This function uses the expected collision count (`events`) and normalizes
    by system `volume` to compute an effective collision rate. It then
    samples from a Poisson distribution to obtain the actual number of
    collisions happening within the current `time_step`.

    Arguments:
        - events : The calculated number of particle pairs that could
          interact.
        - volume : The volume of the simulation space (m³).
        - time_step : The time span (seconds) over which collisions are
          considered.
        - generator : A NumPy random Generator to sample the Poisson
          distribution.

    Returns:
        - The sampled number of coagulation events as an integer.

    Examples:
        ```py
        from numpy.random import default_rng
        rng = default_rng(42)
        collisions = _sample_events(events=5e3, volume=0.1, time_step=0.01,
            generator=rng)
        # collisions might be ~ Poisson( 5e3 / 0.1 * 0.01 ) => Poisson(5)
        ```
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
    """Select valid particle indices in two bins for coagulation events.

    This function tries to choose `events` valid indices from
    `lower_bin` and `upper_bin`, discarding any particles with radius ≤ 0.
    It uses the provided random generator to perform the sampling
    with replacement if needed.

    Arguments:
        - lower_bin : Index of the lower bin to filter particles from.
        - upper_bin : Index of the upper bin to filter particles from.
        - events : Number of events (indices) to sample for each bin.
        - particle_radius : Array of particle radii; only those > 0
          are considered valid.
        - bin_indices : Array of bin labels corresponding to each particle.
        - generator : Random number generator used for index selection.

    Returns:
        - Indices of particles from the lower bin.
        - Indices of particles from the upper bin.

    Examples:
        ```py
        import numpy as np
        rng = np.random.default_rng(123)
        radius = np.array([0.3, 0.1, 0.0, 0.5])
        bins = np.array([0, 0, 1, 1])
        lw_bin, up_bin = random_choice_indices(0, 1, 2, radius, bins, rng)
        # lw_bin -> array of valid picks from bin 0
        # up_bin -> array of valid picks from bin 1
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


def _select_random_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    number_in_bins: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Randomly choose indices within each bin to represent collision partners.

    This function picks `events` indices from the population of the
    `lower_bin` and `upper_bin`, ignoring any radius or event-limit checks
    (those may happen later). The result is two arrays of equal size,
    each containing random picks within the respective bins.

    Arguments:
        - lower_bin : Index for the "smaller" bin.
        - upper_bin : Index for the "larger" bin.
        - events : How many pairs to select.
        - number_in_bins : Array with the count of particles in each bin.
        - generator : Random number generator to draw the indices.

    Returns:
        - An array of size `events` with random picks from `lower_bin`.
        - An array of size `events` with random picks from `upper_bin`.

    Examples:
        ```py
        import numpy as np
        rng = np.random.default_rng(42)
        n_in_bins = np.array([5, 10, 7])
        i_lw, i_up = _select_random_indices(
            lower_bin=0,
            upper_bin=2,
            events=3,
            number_in_bins=n_in_bins,
            generator=rng
        )
        # i_lw -> random indices in [0..4]
        # i_up -> random indices in [0..6]
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


def _bin_to_particle_indices(
    lower_indices: NDArray[np.int64],
    upper_indices: NDArray[np.int64],
    lower_bin: int,
    upper_bin: int,
    bin_indices: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Map bin-relative indices back to absolute positions in the particle
    array.

    This function adjusts the offsets for each bin so that the pairwise
    indices used for collision are mapped onto the actual sorted particle
    array. For instance, if `lower_indices` are all within bin 0, and bin 0
    particles occupy positions [0..9], this method adds that offset to
    each index in `lower_indices`.

    Arguments:
        - lower_indices : Relative indices (local to the bin) of smaller
            particles.
        - upper_indices : Relative indices (local to the bin) of larger
            particles.
        - lower_bin : The bin representing the smaller particles.
        - upper_bin : The bin representing the larger particles.
        - bin_indices : Cumulative offsets to determine where each bin begins.

    Returns:
        - `small_index` : Absolute positions of smaller particles in the
          sorted particle array.
        - `large_index` : Absolute positions of the larger particles in
          the sorted array.

    Examples:
        ```py
        bins = np.array([0, 10, 20])
        lw_rel = np.array([0, 1])
        up_rel = np.array([2, 3])
        # Convert these local indices for bin 1 (start=10) and bin 2 (start=20)
        s_idx, l_idx = _bin_to_particle_indices(lw_rel, up_rel, 1, 2, bins)
        # s_idx -> [10, 11]
        # l_idx -> [22, 23]
    """
    # Get the start index in the particle array for the lower_bin
    start_index_lower_bin = np.searchsorted(bin_indices, lower_bin)
    # Get the start index in the particle array for the upper_bin
    start_index_upper_bin = np.searchsorted(bin_indices, upper_bin)

    # Calculate the actual particle indices for the lower_bin and upper_bin
    small_index = start_index_lower_bin + lower_indices
    large_index = start_index_upper_bin + upper_indices

    return small_index, large_index


def _filter_valid_indices(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
    single_event_counter: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Remove invalid pairs of particles based on radius and optional event
    limit.

    This function checks each pair of `(small_index, large_index)` to ensure
    both have radius > 0. If `single_event_counter` is provided, it further
    enforces that each particle has had < 1 event so far (or you can
    define your own threshold). The pairs failing these checks are removed.

    Arguments:
        - small_index : Indices for the smaller particles in each pair.
        - large_index : Indices for the larger particles in each pair.
        - particle_radius : Array of radii for each particle.
        - single_event_counter : Optional array counting how many events
          each particle has undergone. If provided, only particles with
          counter < 1 pass the filter.

    Returns:
        - Filtered `small_index` with only valid pairs.
        - Filtered `large_index` that corresponds to valid pairs.

    Examples:
        ```py
        r = np.array([0.1, 0.0, 0.08, 0.02])
        c = np.array([0, 0, 0, 0])
        small_i = np.array([0, 1, 2])
        large_i = np.array([3, 0, 1])
        # Filter out pairs with radius <= 0 or event_counter >= 1
        s_valid, l_valid = _filter_valid_indices(
            small_i, large_i, r, single_event_counter=c
        )
        # Indices with r>0 remain in s_valid, l_valid
        ```
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


def _coagulation_events(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    kernel_values: NDArray[np.float64],
    kernel_max: float,
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Stochastically pick which collisions (among possible pairs) actually
    happen.

    This function computes a collision probability for each `(small_index,
    large_index)` pair by taking the ratio of `kernel_values / kernel_max`.
    Next, a random uniform draw decides if each collision occurs.

    Arguments:
        - small_index : Array of indices representing smaller particles.
        - large_index : Array of indices representing larger particles.
        - kernel_values : Collision kernel values for each pair.
        - kernel_max : A maximum kernel value used for normalization.
        - generator : Random generator to compare probabilities vs.
          uniform draws.

    Returns:
        - Filtered `small_index` containing only those that coagulated.
        - Filtered `large_index` containing only those that coagulated.

    Examples:
        ```py
        rng = np.random.default_rng(999)
        s_idx = np.array([0, 1, 2])
        l_idx = np.array([3, 4, 5])
        kv = np.array([0.5, 1.0, 0.1])
        kmax = 1.0
        s_new, l_new = _coagulation_events(s_idx, l_idx, kv, kmax, rng)
        # Each pair has probability kv/kmax => [0.5, 1.0, 0.1]
        # The final s_new, l_new depends on random draws
        ```
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


def _sort_particles(
    particle_radius: NDArray[np.float64],
    particle_concentration: Optional[NDArray[np.float64]] = None,
) -> Tuple[
    NDArray[np.int64], NDArray[np.float64], Optional[NDArray[np.float64]]
]:
    """Sort particle radii (and optionally concentrations) in ascending order.

    The function returns an array of `unsort_indices` that can be used
    to restore the particles to their original order after manipulations.

    Arguments:
        - particle_radius : 1D NumPy array of particle radii.
        - particle_concentration : Optional array of corresponding
          concentrations.

    Returns:
        - `unsort_indices` : Indices to revert sorting to the original order.
        - `sorted_radius` : Sorted array of radii in ascending order.
        - `sorted_concentration` : Sorted array of concentrations, if
          provided; otherwise `None`.

    Examples:
        ```py
        import numpy as np
        r = np.array([0.3, 0.1, 0.5])
        c = np.array([10, 30, 20])
        u_idx, s_r, s_c = _sort_particles(r, c)
        # s_r -> [0.1, 0.3, 0.5]
        # s_c -> [30, 10, 20]
        # u_idx can be used to get them back in [0.3, 0.1, 0.5] order
        ```
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


def _bin_particles(
    particle_radius: NDArray[np.float64],
    radius_bins: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Divide the sorted particle radii into bins and count how many fall into
    each bin.

    This function uses `radius_bins` as edges and assigns each particle
    radius to a bin index via `np.digitize`. The result is (1) a histogram
    with the number of particles in each bin, and (2) an array of per-particle
    bin indices.

    Arguments:
        - particle_radius : Array of sorted particle radii.
        - radius_bins : Edges used to define the bins.

    Returns:
        - `number_in_bins` : Counts of how many radii lie in each bin.
        - `bin_indices` : The bin index assigned to each particle.

    Examples:
        ```py
        import numpy as np
        rad = np.array([1e-9, 1.5e-9, 2e-9, 5e-9])
        bin_edges = np.array([1e-9, 2e-9, 3e-9, 1e-8])
        n_in_bins, bin_idx = _bin_particles(rad, bin_edges)
        # n_in_bins -> [1, 2, 1]
        # bin_idx might be [0, 1, 1, 2]
        ```
    """
    number_in_bins, bins = np.histogram(particle_radius, bins=radius_bins)
    bin_indices = np.digitize(particle_radius, bins, right=True)

    # Handle overflow and underflow bin indices
    bin_indices[bin_indices == len(bins)] = len(bins) - 1
    bin_indices[bin_indices == 0] = 1
    bin_indices -= 1  # Adjust to 0-based indexing

    return number_in_bins, bin_indices


def _get_bin_pairs(
    bin_indices: NDArray[np.int64],
) -> list[Tuple[int, int]]:
    """Produce the list of all unique (binA, binB) pairs using combinations
    with replacement.

    This function is useful when we want to iterate over all bin pairs
    (including binA == binB) for collision computations. The combination
    ensures each pair is returned only once.

    Arguments:
        - bin_indices : Array of bin indices for each particle (though
          only the unique values matter).

    Returns:
        - A list of (lower_bin, upper_bin) pairs covering all unique
          bins in `bin_indices`.

    Examples:
        ```py
        bins = np.array([0, 0, 1, 2, 2])
        pairs = _get_bin_pairs(bins)
        # pairs -> [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        ```
    """
    unique_bins = np.unique(bin_indices)
    return list(combinations_with_replacement(unique_bins, 2))


def _calculate_concentration_in_bins(
    bin_indices: NDArray[np.int64],
    particle_concentration: NDArray[np.float64],
    number_in_bins: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum the particle concentrations in each bin.

    Given per-particle `bin_indices` and `particle_concentration`, this
    function accumulates the total concentration of all particles that
    fall into each bin. The `number_in_bins` array is used mainly for
    shape reference but can also confirm the count of particles.

    Arguments:
        - bin_indices : Array of bin indices for each particle.
        - particle_concentration : 1D array of concentrations matching
          each particle.
        - number_in_bins : Array with the count of particles in each bin.

    Returns:
        - A 1D array whose length is the number of unique bins, containing
          the summed concentration per bin.

    Examples:
        ```py
        import numpy as np
        b_idx = np.array([0, 0, 1, 1, 2])
        conc = np.array([10., 5., 2., 3., 4.])
        n_in_bins = np.array([2, 2, 1])  # might match the bin partition
        bin_c = _calculate_concentration_in_bins(b_idx, conc, n_in_bins)
        # bin_c -> [15., 5., 4.]
        ```
    """
    concentration_in_bins = np.zeros_like(number_in_bins, dtype=np.float64)
    unique_bins = np.unique(bin_indices)

    for unique_bin in unique_bins:
        concentration_in_bins[unique_bin] = np.sum(
            particle_concentration[bin_indices == unique_bin]
        )

    return concentration_in_bins


# pylint: disable=too-many-positional-arguments, too-many-arguments, too-many-locals
def get_super_droplet_coagulation_step(
    particle_radius: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Carry out one time-step of super-droplet-based coagulation.

    This function sorts particles by radius, bins them, and then stochastically
    computes collision events according to the coagulation kernel. It updates
    the particle radii/concentrations, then unsorts them back to the original
    order.

    Arguments:
        - particle_radius : Array of particle radii (m).
        - particle_concentration : Array of per-particle concentration.
        - kernel : 2D matrix of coagulation kernel values, dimension
          ~ len(kernel_radius) × len(kernel_radius).
        - kernel_radius : Array of radius points defining the kernel dimension.
        - volume : System volume or domain size in m³.
        - time_step : The length of this coagulation iteration in seconds.
        - random_generator : Random number generator for sampling collisions.

    Returns:
        - Updated radii array after processing coagulation.
        - Updated concentrations array after processing coagulation.

    Examples:
        ```py
        import numpy as np
        from numpy.random import default_rng
        radius = np.array([1e-9, 2e-9, 5e-9])
        conc = np.array([100., 50., 10.])
        ker_vals = np.ones((3,3))
        ker_r = np.array([1e-9, 2e-9, 5e-9])
        rng = default_rng(42)
        r_new, c_new = get_super_droplet_coagulation_step(
            radius, conc, ker_vals, ker_r, 1e-3, 1.0, rng)
        # r_new, c_new have updated values after one super droplet
        # coagulation step.

    References:
        - E. W. Tedford and L. A. Perugini, "Superdroplet method
          in cloud microphysics simulations," J. Atmos. Sci., 2020.
        - Seinfeld, J. H., & Pandis, S. N. *Atmospheric Chemistry and Physics*,
          Wiley, 2016.
    """
    # Step 1: Sort particles by size and obtain indices to revert sorting later
    unsort_indices, sorted_radius, sorted_concentration = _sort_particles(
        particle_radius=particle_radius,
        particle_concentration=particle_concentration,
    )
    # Step 2: Bin particles by size using the provided kernel radius bins
    number_in_bins, bin_indices = _bin_particles(
        particle_radius=sorted_radius, radius_bins=kernel_radius
    )
    # Step 3: Precompute unique bin pairs for efficient coagulation
    # computations
    pair_indices = _get_bin_pairs(bin_indices=bin_indices)

    # Step 4: Calculate the total concentration of particles within each bin
    concentration_in_bins = _calculate_concentration_in_bins(
        bin_indices=bin_indices,
        particle_concentration=sorted_concentration,  # type: ignore[arg-type]
        number_in_bins=number_in_bins,  # type: ignore[arg-type]
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
        events = _event_pairs(
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            kernel_max=kernel_max,
            number_in_bins=concentration_in_bins,
        )

        # Step 7.3: Sample the number of coagulation events from a
        # Poisson distribution
        num_events = _sample_events(
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
        r_i_indices, r_j_indices = _select_random_indices(
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            events=num_events,
            number_in_bins=number_in_bins,
            generator=random_generator,
        )

        # Step 7.7: Convert bin-relative indices to actual particle indices
        # in the sorted arrays
        indices_i, indices_j = _bin_to_particle_indices(
            lower_indices=r_i_indices,
            upper_indices=r_j_indices,
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            bin_indices=bin_indices,
        )

        # Step 7.8: Filter out invalid particle pairs based on their radii
        # and event counters
        indices_i, indices_j = _filter_valid_indices(
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
        indices_i, indices_j = _coagulation_events(
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
            _super_droplet_update_step(
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
