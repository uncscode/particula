"""Lognormal size distribution properties.
"""

# from typing import Union
from scipy.stats import lognorm
import numpy as np
from numpy.typing import NDArray


def lognormal_pdf_distribution(
    x_values: NDArray[np.float_],
    mode: NDArray[np.float_],
    geometric_standard_deviation: NDArray[np.float_],
    number_of_particles: NDArray[np.float_],
) -> NDArray[np.float_]:
    """
    Probability Density Function for the lognormal distribution of particles
    for varying modes, geometric standard deviations, and numbers of particles,
    across a range of x_values.

    Args:
        x_values: The size interval of the distribution.
        mode: Scales corresponding to the mode in lognormal for different
            modes.
        geometric_standard_deviation: Geometric standard deviations of the
            distribution for different modes.
        number_of_particles: Number of particles for each mode.

    Returns:
        The normalized lognormal distribution of the particles, summed
        across all modes.

    References:
        - [Log-normal Distribution Wikipedia](
            https://en.wikipedia.org/wiki/Log-normal_distribution)
         - [Probability Density Function Wikipedia](
            https://en.wikipedia.org/wiki/Probability_density_function)
    """
    if not (
        x_values.ndim == 1
        and geometric_standard_deviation.shape
        == mode.shape
        == number_of_particles.shape
    ):
        raise ValueError(
            "The shapes of geometric_standard_deviation, "
            "mode, and number_of_particles must match."
        )

    # Calculate PDF for each set of parameters
    distribution = lognorm.pdf(
        x=x_values[:, np.newaxis],
        s=np.log(geometric_standard_deviation),
        scale=mode,
    )

    area = np.trapz(distribution, x=x_values[:, np.newaxis], axis=0)
    scaled_distribution = distribution * (number_of_particles / area)
    return scaled_distribution.sum(axis=1)


def lognormal_pmf_distribution(
    x_values: NDArray[np.float_],
    mode: NDArray[np.float_],
    geometric_standard_deviation: NDArray[np.float_],
    number_of_particles: NDArray[np.float_],
) -> NDArray[np.float_]:
    """
    Probability Mass function for lognormal distribution of particles for
    varying modes, geometric standard deviations, and numbers of particles,
    across a range of x_values.

    Args:
        x_values: The size interval of the distribution.
        mode: Scales corresponding to the mode in lognormal for different
            modes.
        geometric_standard_deviation: Geometric standard deviations of the
            distribution for different modes.
        number_of_particles: Number of particles for each mode.

    Returns:
        The normalized lognormal distribution of the particles, summed
        across all modes.

    References:
        - [Log-normal Distribution Wikipedia](
            https://en.wikipedia.org/wiki/Log-normal_distribution)
        - [Probability Mass Function Wikipedia](
            https://en.wikipedia.org/wiki/Probability_mass_function)
    """
    if not (
        x_values.ndim == 1
        and geometric_standard_deviation.shape
        == mode.shape
        == number_of_particles.shape
    ):
        raise ValueError(
            "The shapes of geometric_standard_deviation, "
            "mode, and number_of_particles must match."
        )

    # Calculate PDF for each set of parameters
    distribution = lognorm.pdf(
        x=x_values[:, np.newaxis],
        s=np.log(geometric_standard_deviation),
        scale=mode,
    )

    bin_sum = np.sum(distribution, axis=0)
    scaled_distribution = distribution * (number_of_particles / bin_sum)
    return scaled_distribution.sum(axis=1)


def lognormal_sample_distribution(
    mode: NDArray[np.float_],
    geometric_standard_deviation: NDArray[np.float_],
    number_of_particles: NDArray[np.float_],
    number_of_samples: int,
) -> NDArray[np.float_]:
    """Sample a Probability Density Function for the lognormal distribution.

    Samples a set of samples (particle) to represent the lognormal distribution
    for varying modes, geometric standard deviations, and numbers of particles,
    across a range of x_values.

    Args:
        mode: Scales corresponding to the mode in lognormal for different
            modes.
        geometric_standard_deviation: Geometric standard deviations of the
            distribution for different modes.
        number_of_particles: Number of particles for each mode.
        number_of_samples: Number of samples to generate.

    Returns:
        The normalized lognormal distribution of the particles, summed
        across all modes.

    References:
        - [Log-normal Distribution Wikipedia](
            https://en.wikipedia.org/wiki/Log-normal_distribution)
         - [Probability Density Function Wikipedia](
            https://en.wikipedia.org/wiki/Probability_density_function)
    """

    # Calculate PDF for each set of parameters
    distribution = lognorm.rvs(
        s=np.log(geometric_standard_deviation),
        scale=mode,
        size=(number_of_samples, number_of_particles.size),
    )

    # Calculate normalized weights and sample accordingly
    weights = number_of_particles / number_of_particles.sum()
    sample_counts = np.ceil(number_of_samples * weights).astype(int)
    samples = np.concatenate(
        [distribution[:count, i] for i, count in enumerate(sample_counts)]
    )

    # Handle over sampling by truncating the samples
    if samples.size > number_of_samples:
        samples = samples[:number_of_samples]

    return samples
