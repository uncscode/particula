"""Calculate Mie optical properties for a size distribution of
spherical particles. With discretization options."""
# pyright: reportReturnType=false, reportAssignmentType=false
# pyright: reportIndexIssue=false
# pyright: reportArgumentType=false, reportOperatorIssue=false
# pylint: disable=too-many-arguments, too-many-locals


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
    Computes Mie coefficients for a spherical particle based on its material
    properties, size, and the properties of the surrounding medium.

    This function leverages the PyMieScatt library to calculate the extinction
    (q_ext), scattering (q_sca), absorption (q_abs) efficiencies, the
    asymmetry factor (g), radiation pressure efficiency (q_pr), backscatter
    efficiency (q_back), and the ratio of backscatter to extinction efficiency
    (q_ratio) for a single sphere under specified conditions.

    This function is optimized with an LRU (Least Recently Used) cache to
    enhance performance by storing up to 100,000 recent calls. The cache
    memorizes the results of expensive function calls and returns the cached
    result when the same inputs occur again, reducing the need to recompute
    these values.

    Args
    ----------
    m_sphere : The complex refractive index of the sphere. For non-absorbing
        material a real number can be provided.
    wavelength : The wavelength of the incident light in nanometers (nm).
    diameter : The diameter of the sphere in nanometers (nm).
    mMedium : The refractive index of the surrounding medium.
        Defaults to 1.0, corresponding to vacuum.

    Returns
    -------
    Tuple[float, float, float, float, float, float, float]
        A tuple containing the calculated Mie efficiencies and parameters:
        q_ext (extinction efficiency), q_sca (scattering efficiency),
        q_abs (absorption efficiency), g (asymmetry factor),
        q_pr (radiation pressure efficiency), q_back (backscatter efficiency),
        and q_ratio (the ratio of backscatter to extinction efficiency).
    """
    return ps.AutoMieQ(m=m_sphere,
                       wavelength=wavelength,
                       diameter=diameter,
                       nMedium=m_medium)


def discretize_mie_parameters(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: Union[float, NDArray[np.float64]],
    base_m_sphere: float = 0.001,
    base_wavelength: float = 1,
    base_diameter: float = 5,
) -> Tuple[
    Union[complex, float],
    float,
    Union[float, list[float]]
]:
    """
    Discretizes the refractive index of the material, the wavelength of
    incident light, and the diameters of particles to enhance the numerical
    stability and performance of Mie scattering calculations. This approach is
    particularly useful for caching Mie scattering computations, as it reduces
    the variability in input parameters leading to a more manageable set of
    unique calculations.

    Parameters
    ----------
    m_sphere : Union[complex, float]
        The complex or real refractive index of the particles. This value is
        discretized to a specified base to reduce the granularity of input
        variations.
    wavelength : float
        The wavelength of incident light in nanometers (nm). It is discretized
        to minimize the variations in computations related to different
        wavelengths.
    diameter : NDArray[np.float64]
        An array of particle diameters in nanometers (nm), each of which is
        discretized to a specified base to standardize the input sizes for
        calculations.
    base_m_sphere : float, optional
        The base value to which the real and imaginary parts of the refractive
        index are rounded. Defaults to 0.001.
    base_wavelength : float, optional
        The base value to which the wavelength is rounded. Defaults to 1 nm.
    base_diameter : float, optional
        The base value to which particle diameters are rounded.
        Defaults to 5 nm.

    Returns
    -------
    Tuple[Union[complex, float], float, NDArray[np.float64]]
    A tuple containing the discretized refractive index (m_sphere), wavelength,
    and diameters (diameter), suitable for use in further Mie scattering
    calculations with potentially improved performance and reduced
    computational overhead.
    """
    m_real = convert.round_arbitrary(
        values=np.real(m_sphere),
        base=base_m_sphere,
        mode='round')
    m_imag = convert.round_arbitrary(
        values=np.imag(m_sphere),
        base=base_m_sphere,
        mode='round')
    # Recombine the discretized real and imaginary parts
    m_discretized = m_real + 1j * m_imag if m_imag != 0 else m_real

    # Discretize the wavelength, assuming nm units
    wavelength_discretized = convert.round_arbitrary(
        values=wavelength,
        base=base_wavelength,
        mode='round')

    # Discretize the particle diameters, assuming nm units
    dp_discretized = convert.round_arbitrary(
        values=diameter,
        base=base_diameter,
        mode='round',
        nonzero_edge=True)

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
    dp: NDArray[np.float64]
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
    """
    Computes bulk optical properties from size dependent efficiency factors for
    the size distribution.

    Parameters
    ----------
    q_ext, q_sca, q_abs, q_pr, q_back, q_ratio, g : NDArray[np.float64]
        Arrays of efficiency factors and the asymmetry factor.
    area : NDArray[np.float64]
        area scaled size distribution.
    extinction_only : bool
        Flag to compute only the extinction coefficient.
    pms : bool
        Flag indicating if probability mass distribution, where sum of
        all bins is total number of particles, is used.
    dp : NDArray[np.float64]
        An array of particle diameters in nanometers.

    Returns
    -------
    Tuple containing bulk optical properties.
    """
    if pms:
        b_ext = np.sum(q_ext * area_dist)
        if extinction_only:
            return b_ext, None, None, None, None, None, None
        b_sca = np.sum(q_sca * area_dist)
        b_abs = b_ext - b_sca
        b_back = np.sum(q_back * area_dist)
        b_ratio = np.sum(q_ratio * area_dist)
        big_g = np.sum(g * q_sca * area_dist) / b_sca \
            if b_sca != 0 else 0
        b_pr = b_ext - big_g * b_sca
    else:  # then pdf so the integral is used
        b_ext = np.trapz(q_ext * area_dist, dp)
        if extinction_only:
            return b_ext, None, None, None, None, None, None
        b_sca = np.trapz(q_sca * area_dist, dp)
        b_abs = b_ext - b_sca
        b_back = np.trapz(q_back * area_dist, dp)
        b_ratio = np.trapz(q_ratio * area_dist, dp)
        big_g = np.trapz(g * q_sca * area_dist, dp) / b_sca \
            if b_sca != 0 else 0
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
    as_dict: bool
) -> Union[dict[str, NDArray[np.float64]], tuple[NDArray[np.float64], ...]]:
    """
    Formats the output results of the Mie scattering calculations.

    Parameters
    ----------
    b_ext, b_sca, b_abs, bigG, b_pr, b_back, b_ratio : float
        Bulk optical properties.
    asDict : bool
        Determines the format of the results.

    Returns
    -------
    Either a dictionary or a tuple of the computed bulk optical properties.
    """
    if as_dict:
        return {
            'b_ext': b_ext,
            'b_sca': b_sca,
            'b_abs': b_abs,
            'G': big_g,
            'b_pr': b_pr,
            'b_back': b_back,
            'b_ratio': b_ratio
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
    truncation_b_sca_multiple: Optional[float] = None
) -> Union[
        NDArray[np.float64],
        dict[str, NDArray[np.float64]],
        Tuple[NDArray[np.float64], ...]]:
    """
    Calculates Mie scattering parameters for a size distribution of spherical
    particles.

    This function computes optical properties such as extinction, scattering,
    absorption coefficients, asymmetry factor, backscatter efficiency, and
    their ratios for a given size distribution of spherical particles. It
    supports various modes of calculation, including discretization of input
    parameters and optional truncation of the scattering efficiency.

    Parameters
    ----------
    m_sphere : Union[complex, float]
        The complex refractive index of the particles. Real values can be used
        for non-absorbing materials.
    wavelength : float
        The wavelength of the incident light in nanometers (nm).
    diameter : NDArray[np.float64]
        An array of particle diameters in nanometers (nm).
    number_per_cm3 : NDArray[np.float64]
        The number distribution of particles per cubic centimeter (#/cm^3).
    n_medium : float, optional
        The refractive index of the medium. Defaults to 1.0 (air or vacuum).
    pms : bool, optional
        Specifies if the size distribution is in probability mass form.
    as_dict : bool, optional
        If True, results are returned as a dictionary. Otherwise, as a tuple.
    extinction_only : bool, optional
        If True, only the extinction coefficient is calculated and returned.
    discretize : bool, optional
        If True, input parameters (m, wavelength, dp) are discretized for
        computation. Defaults to False.
    truncation_calculation : bool, optional
        Enables truncation of the scattering efficiency based on a multiple
        of the backscattering coefficient. Defaults to False.
    truncation_b_sca_multiple : Optional[float], optional
        The multiple of the backscattering coefficient used for truncating the
        scattering efficiency. Required if `truncation_calculation` is True.

    Returns
    -------
    Union[float, Dict[str, float], Tuple[float, ...]]
        Depending on the parameters `asDict` and `extinction_only`, the
        function can return:
        - A single float (extinction coefficient) if `extinction_only` is True.
        - A dictionary of computed optical properties if `asDict` is True.
        - A tuple of computed optical properties.

    Raises
    ------
    ValueError
        If `truncation_calculation` is True but `truncation_b_sca_multiple`
        is not specified.
    """
    # Adjust input parameters for medium's refractive index
    m_sphere /= n_medium
    wavelength /= np.real(n_medium)

    # Ensure inputs are numpy arrays for vectorized operations
    diameter, number_per_cm3 = map(
        lambda x: convert.coerce_type(x, np.ndarray),
        (diameter, number_per_cm3))

    # Initialize arrays for Mie efficiencies and asymmetry factor
    q_ext, q_sca, q_abs, q_pr, q_back, q_ratio, g = (
        np.zeros(diameter.size) for _ in range(7))

    # Calculate area size distribution normalized to inverse megameters
    area_dist = np.pi * (diameter / 2)**2 * number_per_cm3 * 1e-6

    # discretize parameters
    if discretize:
        # Discretize parameters for potentially improved stability/performance
        m_sphere, wavelength, diameter = discretize_mie_parameters(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter
        )

    # Select the appropriate Mie calculation function with discretize flag
    mie_function = discretize_auto_mieq if discretize else ps.AutoMieQ
    # applying discretized parameters if they were set else full MieQ
    for i in range(diameter.size):
        # Perform the calculation
        q_ext[i], q_sca[i], q_abs[i], g[i], q_pr[i], q_back[i], q_ratio[i] = \
            mie_function(
            m_sphere, wavelength, diameter[i], n_medium
        )  # pyright: ignore[reportGeneralTypeIssues]

    # Apply optional truncation to scattering efficiency
    if truncation_calculation:
        if truncation_b_sca_multiple is None:
            raise ValueError(
                "truncation_b_sca_multiple must be specified \
                    for truncation calculation.")
        q_sca *= truncation_b_sca_multiple

    # Compute bulk optical properties based on size distribution and selected
    # mode
    b_ext, b_sca, b_abs, b_pr, b_back, b_ratio, big_g = compute_bulk_optics(
        q_ext, q_sca, q_back, q_ratio,
        g, area_dist, extinction_only, pms, diameter)

    # Return results based on requested format
    if extinction_only:
        return b_ext
    return format_mie_results(b_ext, b_sca, b_abs, big_g,
                              b_pr, b_back, b_ratio, as_dict)
