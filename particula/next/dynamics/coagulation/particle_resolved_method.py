"""Particle resolved method for coagulation.
"""
from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline  # type: ignore

from particula.next.dynamics.coagulation.super_droplet_method import (
    sort_particles,
    bin_particles,
    get_bin_pairs,
    select_random_indices,
    filter_valid_indices,
    sample_events,
    event_pairs,
    bin_to_particle_indices,
    coagulation_events,
)


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


# pylint: disable=too-many-arguments, too-many-locals
def particle_resolved_coagulation_step(
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64]
]:
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
        Tuple: Updated particle radii, and arrays representing the loss and
            gain in particle counts due to coagulation events.
    """

    # Step 1: Sort particles by size and obtain indices to revert sorting later
    unsort_indices, sorted_radius, _ = sort_particles(
        particle_radius=particle_radius,
    )

    # Step 2: Bin particles by size using the provided kernel radius bins
    number_in_bins, bin_indices = bin_particles(
        particle_radius=sorted_radius, radius_bins=kernel_radius
    )

    # Step 3: Precompute unique bin pairs for efficient coagulation
    # computations
    pair_indices = get_bin_pairs(bin_indices=bin_indices)

    # Step 4: Initialize a bivariate spline for interpolating kernel
    # values between bin radii
    interp_kernel = RectBivariateSpline(
        x=kernel_radius, y=kernel_radius, z=kernel
    )

    # Initialize loss and gain arrays
    loss = np.zeros_like(particle_radius, dtype=np.float64)
    gain = np.zeros_like(particle_radius, dtype=np.float64)
    loss_index = np.zeros_like(particle_radius, dtype=np.int64)
    gain_index = np.zeros_like(particle_radius, dtype=np.int64)

    # Iterate over each bin pair to calculate potential coagulation events
    for lower_bin, upper_bin in pair_indices:
        # Retrieve the maximum kernel value for the current bin pair
        kernel_max = kernel[lower_bin, upper_bin + 1]

        # Determine potential coagulation events between particles in these
        # bins
        particle_events = event_pairs(
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            kernel_max=kernel_max,
            number_in_bins=number_in_bins,
        )

        # Sample the number of coagulation events from a Poisson distribution
        num_particle_events = sample_events(
            events=particle_events,
            volume=volume,
            time_step=time_step,
            generator=random_generator,
        )

        # Skip to the next bin pair if no events are expected
        if num_particle_events == 0:
            continue

        # Randomly select indices of particles involved in the coagulation
        # events within the current bins
        lower_indices, upper_indices = select_random_indices(
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            events=num_particle_events,
            number_in_bins=number_in_bins,
            generator=random_generator,
        )

        # Convert bin-relative indices to actual particle indices in the
        # sorted arrays
        small_index, large_index = bin_to_particle_indices(
            lower_indices=lower_indices,
            upper_indices=upper_indices,
            lower_bin=lower_bin,
            upper_bin=upper_bin,
            bin_indices=bin_indices,
        )

        # Filter out invalid particle pairs based on their radii and event
        # counters
        small_index, large_index = filter_valid_indices(
            small_index=small_index,
            large_index=large_index,
            particle_radius=particle_radius,
        )

        # Skip to the next bin pair if no valid indices remain after filtering
        if small_index.size == 0:
            continue

        # Interpolate kernel values for the selected particle pairs
        kernel_values = interp_kernel.ev(
            particle_radius[small_index], particle_radius[large_index]
        )

        # Determine which coagulation events actually occur based on
        # interpolated kernel probabilities
        small_index, large_index = coagulation_events(
            small_index=small_index,
            large_index=large_index,
            kernel_values=kernel_values,
            kernel_max=kernel_max,
            generator=random_generator,
        )

        # Evaluate the coagulation events and update particle radii,
        # loss, and gain
        particle_radius, loss, gain = particle_resolved_update_step(
            particle_radius=particle_radius,
            loss=loss,
            gain=gain,
            small_index=small_index,
            large_index=large_index,
        )
        loss_index[small_index] = small_index
        gain_index[small_index] = large_index

    # Unsort the particle radii and loss/gain arrays to match the original
    # order
    particle_radius = particle_radius[unsort_indices]
    loss = loss[unsort_indices]
    gain = gain[unsort_indices]
    loss_gain_index = np.column_stack(
        [loss_index, gain_index])
    loss_gain_index = loss_gain_index[unsort_indices]
    # remove particles with zero radius
    loss_gain_index = loss_gain_index[loss_gain_index.sum(axis=1) > 0]

    return particle_radius, loss, gain, loss_gain_index
