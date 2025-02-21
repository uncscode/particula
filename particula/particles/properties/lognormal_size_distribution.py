"""Lognormal size distribution properties."""

# from typing import Union
from scipy.stats import lognorm
import numpy as np
from numpy.typing import NDArray

from particula.util.convert import distribution_convert_pdf_pms


def get_lognormal_pdf_distribution(
    x_values: NDArray[np.float64],
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute a lognormal probability density function (PDF) for given modes.

    This function superimposes multiple lognormal PDFs, each with its own mode,
    geometric standard deviation, and particle count. It then returns their sum
    across the provided x_values.

    Mathematically, for each mode i:

    - PDFᵢ(x) = (1 / [x · ln(gsdᵢ) · √(2π)]) ×
                 exp(- [ln(x) - ln(modeᵢ)]² / [2 · (ln(gsdᵢ))² ])

    Arguments:
        - x_values : 1D array of the size points at which the PDF is evaluated.
        - mode : Array of lognormal mode (scale) values for each mode.
        - geometric_standard_deviation : Array of GSD values for each mode.
        - number_of_particles : Number of particles in each mode.

    Returns:
        - 1D array of the total PDF values summed across all modes.

    Examples:
        ```py title="Example"
        import numpy as np
        from particula.particles.properties.lognormal_size_distribution import get_lognormal_pdf_distribution

        x_vals = np.linspace(1e-9, 1e-6, 100)
        pdf = get_lognormal_pdf_distribution(
            x_values=x_vals,
            mode=np.array([5e-8, 1e-7]),
            geometric_standard_deviation=np.array([1.5, 2.0]),
            number_of_particles=np.array([1e9, 5e9])
        )
        print(pdf[:10])
        # Output: [...]
        ```

    References:
        - [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
        - [Probability Density Function Wikipedia](https://en.wikipedia.org/wiki/Probability_density_function)
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

    area = np.trapezoid(distribution, x=x_values[:, np.newaxis], axis=0)
    area[area == 0] = np.nan

    scaled_distribution = distribution * (number_of_particles / area)

    return np.nansum(scaled_distribution, axis=1)


def get_lognormal_pmf_distribution(
    x_values: NDArray[np.float64],
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute a lognormal probability mass function (PMF) for given modes.

    This function first calculates the lognormal PDF using
    get_lognormal_pdf_distribution(), then converts it to a PMF by
    integrating (or summing) over x_values. The result reflects discrete mass
    (probability) distribution across the given size points.

    Arguments:
        - x_values : 1D array of size points at which the PMF is evaluated.
        - mode : Array of lognormal mode (scale) values for each mode.
        - geometric_standard_deviation : Array of GSD values for each mode.
        - number_of_particles : Number of particles in each mode.

    Returns:
        - 1D array of the total PMF values summed across all modes.

    Examples:
        ```py title="Example"
        import numpy as np
        from particula.particles.properties.lognormal_size_distribution import get_lognormal_pmf_distribution

        x_vals = np.linspace(1e-9, 1e-6, 100)
        pmf = get_lognormal_pmf_distribution(
            x_values=x_vals,
            mode=np.array([5e-8, 1e-7]),
            geometric_standard_deviation=np.array([1.5, 2.0]),
            number_of_particles=np.array([1e9, 5e9])
        )
        print(pmf[:10])
        # Output: [...]
        ```

    References:
        - [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
        - [Probability Mass Function Wikipedia](https://en.wikipedia.org/wiki/Probability_mass_function)
    """

    distribution_pdf = get_lognormal_pdf_distribution(
        x_values=x_values,
        mode=mode,
        geometric_standard_deviation=geometric_standard_deviation,
        number_of_particles=number_of_particles,
    )

    # convert PDF to PMF
    distribution_pmf = distribution_convert_pdf_pms(
        x_array=x_values,
        distribution=distribution_pdf,
        to_pdf=False,
    )

    # check total number of particles
    distribution_pmf_sum = np.sum(distribution_pmf)
    return distribution_pmf * np.divide(
        np.sum(number_of_particles),
        distribution_pmf_sum,
        out=np.ones_like(distribution_pmf_sum),
        where=distribution_pmf_sum != 0,
    )


def get_lognormal_sample_distribution(
    mode: NDArray[np.float64],
    geometric_standard_deviation: NDArray[np.float64],
    number_of_particles: NDArray[np.float64],
    number_of_samples: int,
) -> NDArray[np.float64]:
    """
    Generate random samples from a lognormal distribution for given modes.

    This function uses scipy.stats.lognorm.rvs() to draw samples for each mode,
    with a specified scale (mode) and shape (GSD). The total samples are then
    combined according to the relative number of particles in each mode.

    Arguments:
        - mode : Array of lognormal mode (scale) values for each mode.
        - geometric_standard_deviation : Array of GSD values for each mode.
        - number_of_particles : Number of particles for each mode.
        - number_of_samples : Total number of random samples to generate.

    Returns:
        - 1D array of sampled particle sizes, combining all modes.

    Examples:
        ```py title="Example"
        import numpy as np
        from particula.particles.properties.lognormal_size_distribution import get_lognormal_sample_distribution

        samples = get_lognormal_sample_distribution(
            mode=np.array([5e-8, 1e-7]),
            geometric_standard_deviation=np.array([1.5, 2.0]),
            number_of_particles=np.array([1e9, 5e9]),
            number_of_samples=10000
        )
        print(samples[:10])
        # Output: [...]
        ```

    References:
        - [Log-normal Distribution Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)
        - [Probability Density Function Wikipedia](https://en.wikipedia.org/wiki/Probability_density_function)
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
