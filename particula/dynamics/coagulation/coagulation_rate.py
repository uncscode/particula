"""Coagulation rate calculations for particle populations.

This module defines discrete and continuous ways (via summation or
integration) to compute the gain and loss terms in coagulation
processes. Each function isolates specific calculation details,
allowing for easier testing and flexibility in usage.

References:
    - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
      physics, Chapter 13, Equation 13.61.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RectBivariateSpline  # type: ignore


def get_coagulation_loss_rate_discrete(
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the coagulation loss rate via a discrete summation approach.

    This function computes the loss rate of particles from collisions by
    summing over all size classes. The equation is:

    - loss_rate = ΣᵢΣⱼ [kernel(i, j) × concentration(i) × concentration(j)]

    Arguments:
        - concentration : The distribution of particles.
        - kernel : The coagulation kernel matrix (NDArray[np.float64]).

    Returns:
        - The coagulation loss rate (float or NDArray[np.float64]).

    Examples:
        ```py
        import numpy as np
        import particula as par

        conc = np.array([1.0, 2.0, 3.0])
        kern = np.ones((3, 3))
        loss = par.dynamics.get_coagulation_loss_rate_discrete(conc, kern)
        print(loss)
        # Example output: 36.0
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Chapter 13, Equation 13.61.
    """
    return np.sum(kernel * np.outer(concentration, concentration), axis=0)


def get_coagulation_gain_rate_discrete(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the coagulation gain rate (using a quasi-continuous approach).

    Though named "discrete," this function converts the discrete distribution
    to a PDF and uses interpolation (RectBivariateSpline) to approximate the
    gain term. The concept is:

    - gain_rate(r) = ∫ kernel(r, r') × PDF(r) × PDF(r') dr'
      (implemented via numeric integration)

    Arguments:
        - radius : The particle radius array [m].
        - concentration : The particle distribution.
        - kernel : Coagulation kernel matrix.

    Returns:
        - The coagulation gain rate, matched to the shape of radius.

    Examples:
        ```py
        import numpy as np
        import particula as par

        r = np.array([1e-7, 2e-7, 3e-7])
        conc = np.array([1.0, 0.5, 0.2])
        kern = np.ones((3, 3)) * 1e-9

        gain_val = par.dynamics.get_coagulation_gain_rate_discrete(
            r, conc, kern
        )
        print(gain_val)
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Chapter 13, Equation 13.61.
    """
    # Calculate bin widths (delta_x_array)
    # Handle Union type: ensure radius is an array for indexing
    if not isinstance(radius, np.ndarray):
        radius = np.asarray([radius])
    if not isinstance(concentration, np.ndarray):
        concentration = np.asarray([concentration])

    delta_x_array = np.diff(radius, append=2 * radius[-1] - radius[-2])

    # Convert concentration to a probability density function (PDF)
    concentration_pdf = concentration / delta_x_array

    # Prepare interpolation for continuous distribution
    interp = RectBivariateSpline(
        x=radius,
        y=radius,
        z=kernel * np.outer(concentration_pdf, concentration_pdf),
    )

    # Define dpd and dpi for integration
    # integration variable
    dpd = np.linspace(0, radius / 2 ** (1 / 3), radius.size)
    # adjusted for broadcasting
    dpi = (np.transpose(radius) ** 3 - dpd**3) ** (1 / 3)

    # Compute gain using numerical integration
    gain = radius**2 * np.trapezoid(interp.ev(dpd, dpi) / dpi**2, dpd, axis=0)  # type: ignore

    # Convert back to original scale (from PDF to PMF)
    return gain * delta_x_array


def get_coagulation_loss_rate_continuous(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the coagulation loss rate via continuous integration.

    This method integrates the product of kernel and concentration over
    the radius grid. The equation is:

    - loss_rate(r) = concentration(r) × ∫ kernel(r, r') × concentration(r') dr'

    Arguments:
        - radius : The particle radius array [m].
        - concentration : The particle distribution.
        - kernel : Coagulation kernel matrix (NDArray[np.float64]).

    Returns:
        - The coagulation loss rate.

    Examples:
        ```py
        import numpy as np
        import particula as par

        r = np.array([1e-7, 2e-7, 3e-7])
        conc = np.array([1.0, 0.5, 0.2])
        kern = np.ones((3, 3)) * 1e-9

        loss_cont = par.dynamics.get_coagulation_loss_rate_continuous(
            r, conc, kern
        )
        print(loss_cont)
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Chapter 13, Equation 13.61.
    """
    # concentration (n,) and kernel (n,n)
    # Cast result to handle numpy return type
    # Using type: ignore for np.trapezoid return type compatibility
    return concentration * np.trapezoid(y=kernel * concentration, x=radius)  # type: ignore[operator,return-value]


def get_coagulation_gain_rate_continuous(
    radius: Union[float, NDArray[np.float64]],
    concentration: Union[float, NDArray[np.float64]],
    kernel: NDArray[np.float64],
) -> Union[float, NDArray[np.float64]]:
    """Calculate the coagulation gain rate via continuous integration.

    This function converts the distribution to a continuous form, then
    uses RectBivariateSpline to interpolate and integrate:

    - gain_rate(r) = ∫ kernel(r, r') × concentration(r) × concentration(r') dr'

    Arguments:
        - radius : The particle radius array [m].
        - concentration : The particle distribution.
        - kernel : Coagulation kernel matrix.

    Returns:
        - The coagulation gain rate, in the shape of radius.

    Examples:
        ```py
        import numpy as np
        import particula as par

        r = np.array([1e-7, 2e-7, 3e-7])
        conc = np.array([1.0, 0.5, 0.2])
        kern = np.ones((3, 3)) * 1e-9

        gain_cont = par.dynamics.get_coagulation_gain_rate_continuous(
            r, conc, kern
        )
        print(gain_cont)
        ```

    References:
        - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
          physics, Chapter 13, Equation 13.61.
    """
    # continuous distribution, kernel (n,n)
    # outer replaces, concentration * np.transpose([concentration])
    interp = RectBivariateSpline(
        x=radius, y=radius, z=kernel * np.outer(concentration, concentration)
    )

    dpd = np.linspace(0, radius / 2 ** (1 / 3), radius.size)  # type: ignore
    dpi = (np.transpose(radius) ** 3 - dpd**3) ** (1 / 3)

    return radius**2 * np.trapezoid(
        interp.ev(dpd, dpi) / dpi**2,  # type: ignore
        dpd,
        axis=0,  # type: ignore
    )
