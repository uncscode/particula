"""Calculate Mie optical properties for a size distribution of
spherical particles. With discretization options."""

# pyright: reportReturnType=false, reportAssignmentType=false
# pyright: reportIndexIssue=false
# pyright: reportArgumentType=false, reportOperatorIssue=false
# pylint: disable=too-many-positional-arguments, too-many-arguments,
# pylint: disable=too-many-locals


from typing import Union, Tuple, Optional
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
import PyMieScatt as ps
from particula.util import convert


@lru_cache(maxsize=100000)
def discretize_auto_mieq(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: float,
    m_medium: float = 1.0,
) -> Tuple[float, ...]:
    """
    Compute Mie coefficients for a spherical particle based on its material
    properties, size, and the properties of the surrounding medium.

    This function uses the PyMieScatt library to calculate various Mie
    efficiencies and parameters for a single sphere, including extinction
    efficiency (q_ext), scattering efficiency (q_sca), absorption efficiency
    (q_abs), asymmetry factor (g), radiation pressure efficiency (q_pr),
    backscatter efficiency (q_back), and the ratio of backscatter to
    extinction efficiency (q_ratio).

    The function is optimized with an LRU (Least Recently Used) cache,
    which stores up to 100,000 recent computations to improve performance
    by avoiding repeated calculations for the same inputs.

    Arguments:
        m_sphere: The complex refractive index of the sphere. A real
            number can be provided for non-absorbing materials.
        wavelength: The wavelength of the incident light in nanometers (nm).
        diameter: The diameter of the sphere in nanometers (nm).
        m_medium: The refractive index of the surrounding medium.
            Default is 1.0, corresponding to vacuum.

    Returns:
        Tuple:
            - q_ext, Extinction efficiency.
            - q_sca, Scattering efficiency.
            - q_abs, Absorption efficiency.
            - g, Asymmetry factor.
            - q_pr, Radiation pressure efficiency.
            - q_back, Backscatter efficiency.
            - q_ratio, Ratio of backscatter to extinction efficiency.
    """
    return ps.AutoMieQ(
        m=m_sphere, wavelength=wavelength, diameter=diameter, nMedium=m_medium
    )


def discretize_mie_parameters(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: Union[float, NDArray[np.float64]],
    base_m_sphere: float = 0.001,
    base_wavelength: float = 1,
    base_diameter: float = 5,
) -> Tuple[Union[complex, float], float, Union[float, list[float]]]:
    """
    Discretize the refractive index, wavelength, and diameters for Mie
    scattering calculations.

    This function improves numerical stability and performance by discretizing
    the refractive index of the material, the wavelength of incident light,
    and the diameters of particles. Discretization reduces the variability
    in input parameters, making Mie scattering computations more efficient
    by creating a more manageable set of unique calculations.

    Arguments:
        m_sphere: The complex or real refractive index of the particles.
            This value is discretized to a specified base to reduce
            input variability.
        wavelength: The wavelength of incident light in nanometers (nm),
            discretized to minimize variations in related computations.
        diameter: The particle diameter or array of diameters in nanometers
            (nm), discretized to a specified base to standardize input sizes
            for calculations.
        base_m_sphere: Optional; the base value to which the real and
            imaginary parts of the refractive index are rounded.
            Default is 0.001.
        base_wavelength: Optional; the base value to which the wavelength is
            rounded. Default is 1 nm.
        base_diameter: Optional; the base value to which particle diameters
            are rounded. Default is 5 nm.

    Returns:
        Tuple:
            - The discretized refractive index (m_sphere).
            - The discretized wavelength.
            - The discretized diameter or array of diameters, suitable for use
                in Mie scattering calculations with potentially improved
                performance and reduced computational overhead.
    """
    m_real = convert.round_arbitrary(
        values=np.real(m_sphere), base=base_m_sphere, mode="round"
    )
    m_imag = convert.round_arbitrary(
        values=np.imag(m_sphere), base=base_m_sphere, mode="round"
    )
    # Recombine the discretized real and imaginary parts
    m_discretized = m_real + 1j * m_imag if m_imag != 0 else m_real

    # Discretize the wavelength, assuming nm units
    wavelength_discretized = convert.round_arbitrary(
        values=wavelength, base=base_wavelength, mode="round"
    )

    # Discretize the particle diameters, assuming nm units
    dp_discretized = convert.round_arbitrary(
        values=diameter, base=base_diameter, mode="round", nonzero_edge=True
    )

    return m_discretized, wavelength_discretized, dp_discretized


def compute_bulk_optics(
    q_ext: NDArray[np.float64],
    q_sca: NDArray[np.float64],
    q_back: NDArray[np.float64],
    q_ratio: NDArray[np.float64],
    g: NDArray[np.float64],
    area_dist: NDArray[np.float64],
    extinction_only: bool,
    pms: bool,
    dp: NDArray[np.float64],
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
    """
    Compute bulk optical properties from size-dependent efficiency factors for
    a size distribution.

    This function calculates various bulk optical properties such as
    extinction, scattering, and backscattering coefficients based on the size
    distribution and corresponding efficiency factors.

    Arguments:
        q_ext: Array of extinction efficiency factors.
        q_sca: Array of scattering efficiency factors.
        q_back: Array of backscatter efficiency factors.
        q_ratio: Array of backscatter-to-extinction ratio efficiency factors.
        g: Array of asymmetry factors.
        area_dist: Area-scaled size distribution array.
        extinction_only: Flag indicating whether to compute only the
            extinction coefficient.
        pms: Flag indicating whether a probability mass distribution is used,
            where the sum of all bins
            represents the total number of particles.
        dp: Array of particle diameters in nanometers.

    Returns:
        If `extinction_only` is True, returns an array with the bulk
        extinction coefficient. Otherwise, returns a tuple containing the
        bulk optical properties, including extinction, scattering, and
        backscattering coefficients, and possibly others depending on
        the input flags.
    """
    if pms:
        b_ext = np.sum(q_ext * area_dist)
        if extinction_only:
            return b_ext, None, None, None, None, None, None
        b_sca = np.sum(q_sca * area_dist)
        b_abs = b_ext - b_sca
        b_back = np.sum(q_back * area_dist)
        b_ratio = np.sum(q_ratio * area_dist)
        big_g = np.sum(g * q_sca * area_dist) / b_sca if b_sca != 0 else 0
        b_pr = b_ext - big_g * b_sca
    else:  # then pdf so the integral is used
        b_ext = np.trapz(q_ext * area_dist, dp)
        if extinction_only:
            return b_ext, None, None, None, None, None, None
        b_sca = np.trapz(q_sca * area_dist, dp)
        b_abs = b_ext - b_sca
        b_back = np.trapz(q_back * area_dist, dp)
        b_ratio = np.trapz(q_ratio * area_dist, dp)
        big_g = (
            np.trapz(g * q_sca * area_dist, dp) / b_sca if b_sca != 0 else 0
        )
        b_pr = b_ext - big_g * b_sca
    return b_ext, b_sca, b_abs, b_pr, b_back, b_ratio, big_g


def format_mie_results(
    b_ext: NDArray[np.float64],
    b_sca: NDArray[np.float64],
    b_abs: NDArray[np.float64],
    big_g: NDArray[np.float64],
    b_pr: NDArray[np.float64],
    b_back: NDArray[np.float64],
    b_ratio: NDArray[np.float64],
    as_dict: bool,
) -> Union[dict[str, NDArray[np.float64]], tuple[NDArray[np.float64], ...]]:
    """
    Format the output results of the Mie scattering calculations.

    Arguments:
        b_ext: Array of bulk extinction coefficients.
        b_sca: Array of bulk scattering coefficients.
        b_abs: Array of bulk absorption coefficients.
        big_g: Array of asymmetry factors (g).
        b_pr: Array of bulk radiation pressure efficiencies.
        b_back: Array of bulk backscattering coefficients.
        b_ratio: Array of backscatter-to-extinction ratios.
        as_dict: Flag to determine if the results should be returned as a
            dictionary.

    Returns:
        (dict, Tuple):
            - If `as_dict` is True, returns a dictionary with the bulk optical
                properties.
            - If `as_dict` is False, returns a tuple of the bulk
                optical properties in the following order,
                (b_ext, b_sca, b_abs, big_g, b_pr, b_back, b_ratio).
    """
    if as_dict:
        return {
            "b_ext": b_ext,
            "b_sca": b_sca,
            "b_abs": b_abs,
            "G": big_g,
            "b_pr": b_pr,
            "b_back": b_back,
            "b_ratio": b_ratio,
        }
    return b_ext, b_sca, b_abs, big_g, b_pr, b_back, b_ratio


def mie_size_distribution(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: NDArray[np.float64],
    number_per_cm3: NDArray[np.float64],
    n_medium: float = 1.0,
    pms: bool = True,
    as_dict: bool = False,
    extinction_only: bool = False,
    discretize: bool = False,
    truncation_calculation: bool = False,
    truncation_b_sca_multiple: Optional[float] = None,
) -> Union[
    NDArray[np.float64],
    dict[str, NDArray[np.float64]],
    Tuple[NDArray[np.float64], ...],
]:
    """
    Calculate Mie scattering parameters for a size distribution of spherical
    particles.

    This function computes optical properties such as extinction, scattering,
    absorption coefficients, asymmetry factor, backscatter efficiency, and
    their ratios for a given size distribution of spherical particles. It
    supports various modes of calculation, including discretization of input
    parameters and optional truncation of the scattering efficiency.

    Parameters:
        m_sphere: The complex refractive index of the particles. Real values
            can be used for non-absorbing materials.
        wavelength: The wavelength of the incident light in nanometers (nm).
        diameter: An array of particle diameters in nanometers (nm).
        number_per_cm3: The number distribution of particles per cubic
            centimeter (#/cm^3).
        n_medium: The refractive index of the medium. Defaults to 1.0
            (air or vacuum).
        pms: Specifies if the size distribution is in probability mass form.
            Default is True.
        as_dict: If True, results are returned as a dictionary. Otherwise,
            as a tuple. Default is False.
        extinction_only: If True, only the extinction coefficient is
            calculated and returned. Default is False.
        discretize: If True, input parameters (m_sphere, wavelength, diameter)
            are discretized for computation. Default is False.
        truncation_calculation: Enables truncation of the scattering
            efficiency based on a multiple of the backscattering coefficient.
            Default is False.
        truncation_b_sca_multiple: The multiple of the backscattering
            coefficient used for truncating the scattering efficiency.
            Required if `truncation_calculation` is True.

    Returns:
        (NDArray, dict, Tuple):
            - An array of extinction coefficients if `extinction_only` is True.
            - A dictionary of computed optical properties if `as_dict` is True.
            - A tuple of computed optical properties otherwise.

    Raises:
        ValueError: If `truncation_calculation` is True but
            `truncation_b_sca_multiple` is not specified.
    """
    # Adjust input parameters for medium's refractive index
    m_sphere /= n_medium
    wavelength /= np.real(n_medium)

    # Ensure inputs are numpy arrays for vectorized operations
    diameter, number_per_cm3 = map(
        lambda x: convert.coerce_type(x, np.ndarray),
        (diameter, number_per_cm3),
    )

    # Initialize arrays for Mie efficiencies and asymmetry factor
    q_ext, q_sca, q_abs, q_pr, q_back, q_ratio, g = (
        np.zeros(diameter.size) for _ in range(7)
    )

    # Calculate area size distribution normalized to inverse megameters
    area_dist = np.pi * (diameter / 2) ** 2 * number_per_cm3 * 1e-6

    # discretize parameters
    if discretize:
        # Discretize parameters for potentially improved stability/performance
        m_sphere, wavelength, diameter = discretize_mie_parameters(
            m_sphere=m_sphere, wavelength=wavelength, diameter=diameter
        )

    # Select the appropriate Mie calculation function with discretize flag
    mie_function = discretize_auto_mieq if discretize else ps.AutoMieQ
    # applying discretized parameters if they were set else full MieQ
    for i in range(diameter.size):
        # Perform the calculation
        q_ext[i], q_sca[i], q_abs[i], g[i], q_pr[i], q_back[i], q_ratio[i] = (
            mie_function(m_sphere, wavelength, diameter[i], n_medium)
        )  # pyright: ignore[reportGeneralTypeIssues]

    # Apply optional truncation to scattering efficiency
    if truncation_calculation:
        if truncation_b_sca_multiple is None:
            raise ValueError(
                "truncation_b_sca_multiple must be specified \
                    for truncation calculation."
            )
        q_sca *= truncation_b_sca_multiple

    # Compute bulk optical properties based on size distribution and selected
    # mode
    b_ext, b_sca, b_abs, b_pr, b_back, b_ratio, big_g = compute_bulk_optics(
        q_ext,
        q_sca,
        q_back,
        q_ratio,
        g,
        area_dist,
        extinction_only,
        pms,
        diameter,
    )

    # Return results based on requested format
    if extinction_only:
        return b_ext
    return format_mie_results(
        b_ext, b_sca, b_abs, big_g, b_pr, b_back, b_ratio, as_dict
    )
