"""Functions for processing optical data."""
# linting disabled until reformatting of this file
# pyright: reportReturnType=false, reportAssignmentType=false
# pyright: reportIndexIssue=false
# pyright: reportArgumentType=false, reportOperatorIssue=false
# ***pytype: skip-file
# **flake8: noqa
# pylint: disable=too-many-arguments, too-many-locals

from typing import Union, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from particula.data.stream import Stream

import PyMieScatt as ps
from scipy.integrate import trapz
from functools import lru_cache
from tqdm import tqdm
from scipy.optimize import fminbound
from particula.util import convert


@lru_cache(maxsize=100000)
def discretize_AutoMieQ(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: float,
    mMedium: float = 1.0,
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
                       nMedium=mMedium)


def discretize_Mie_parameters(
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
    aSDn: NDArray[np.float64],
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
    aSDn : NDArray[np.float64]
        Number size distribution.
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
        b_ext = np.sum(q_ext * aSDn)
        if extinction_only:
            return b_ext
        b_sca = np.sum(q_sca * aSDn)
        b_abs = b_ext - b_sca
        b_back = np.sum(q_back * aSDn)
        b_ratio = np.sum(q_ratio * aSDn)
        bigG = np.sum(g * q_sca * aSDn) / b_sca \
            if b_sca != 0 else 0
        b_pr = b_ext - bigG * b_sca
    else:  # then pdf so the integral is used
        b_ext = np.trapz(q_ext * aSDn, dp)
        if extinction_only:
            return b_ext
        b_sca = np.trapz(q_sca * aSDn, dp)
        b_abs = b_ext - b_sca
        b_back = np.trapz(q_back * aSDn, dp)
        b_ratio = np.trapz(q_ratio * aSDn, dp)
        bigG = np.trapz(g * q_sca * aSDn, dp) / b_sca \
            if b_sca != 0 else 0
        b_pr = b_ext - bigG * b_sca
    return b_ext, b_sca, b_abs, b_pr, b_back, b_ratio, bigG


def format_Mie_results(
    b_ext: NDArray[np.float64],
    b_sca: NDArray[np.float64],
    b_abs: NDArray[np.float64],
    bigG: NDArray[np.float64],
    b_pr: NDArray[np.float64],
    b_back: NDArray[np.float64],
    b_ratio: NDArray[np.float64],
    asDict: bool
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
    if asDict:
        return {
            'b_ext': b_ext,
            'b_sca': b_sca,
            'b_abs': b_abs,
            'G': bigG,
            'b_pr': b_pr,
            'b_back': b_back,
            'b_ratio': b_ratio
        }
    else:
        return b_ext, b_sca, b_abs, bigG, b_pr, b_back, b_ratio


def mie_size_distribution(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: NDArray[np.float64],
    number_per_cm3: NDArray[np.float64],
    n_medium: float = 1.0,
    pms: bool = True,
    asDict: bool = False,
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
    asDict : bool, optional
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
        lambda x: convert.coerce_type(x, NDArray[np.float64]),
        (diameter, number_per_cm3))

    # Initialize arrays for Mie efficiencies and asymmetry factor
    q_ext, q_sca, q_abs, q_pr, q_back, q_ratio, g = (
        np.zeros(diameter.size) for _ in range(7))

    # Calculate area size distribution normalized to inverse megameters
    aSDn = np.pi * (diameter / 2)**2 * number_per_cm3 * 1e-6

    # discretize parameters
    if discretize:
        # Discretize parameters for potentially improved stability/performance
        m_sphere, wavelength, diameter = discretize_Mie_parameters(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter
        )

    # Select the appropriate Mie calculation function with discretize flag
    mie_function = discretize_AutoMieQ if discretize else ps.AutoMieQ
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
    b_ext, b_sca, b_abs, b_pr, b_back, b_ratio, bigG = compute_bulk_optics(
        q_ext, q_sca, q_back, q_ratio,
        g, aSDn, extinction_only, pms, diameter)

    # Return results based on requested format
    if extinction_only:
        return b_ext
    return format_Mie_results(b_ext, b_sca, b_abs, bigG,
                              b_pr, b_back, b_ratio, asDict)


def extinction_ratio_wet_dry(
    kappa: Union[float, NDArray[np.float64]],
    number_per_cm3: NDArray[np.float64],
    diameters: NDArray[np.float64],
    water_activity_sizer: NDArray[np.float64],
    water_activity_dry: NDArray[np.float64],
    water_activity_wet: NDArray[np.float64],
    refractive_index_dry: Union[complex, float] = 1.45,
    water_refractive_index: Union[complex, float] = 1.33,
    wavelength: float = 450,
    discretize_Mie: bool = True,
    return_coefficients: bool = False,
    return_all_optics: bool = False,
) -> Union[float, Tuple[NDArray, NDArray]]:
    """
    Calculates the extinction ratio between wet and dry aerosols, considering
    water uptake through kappa. This function uses Mie theory to determine the
    optical properties of aerosols with varying water content, allowing for
    analysis of hygroscopic growth and its impact on aerosol optical
    characteristics.

    Parameters
    ----------
    kappa : Union[float, NDArray[np.float64]]
        Hygroscopicity parameter, defining water uptake ability of particles.
    number_per_cm3 : NDArray[np.float64]
        Number concentration of particles per cubic centimeter for each size
        bin.
    diameters : NDArray[np.float64]
        Diameters of particles in nanometers for each size bin.
    water_activity_sizer : NDArray[np.float64]
        Water activity of the aerosol size distribution.
    water_activity_dry : NDArray[np.float64]
        Water activity for the calculation of 'dry' aerosol properties.
    water_activity_wet : NDArray[np.float64]
        Water activity for the calculation of 'wet' aerosol properties.
    refractive_index_dry : Union[complex, float, np.float16], optional
        Refractive index of the dry aerosol particles.
    water_refractive_index : Union[complex, float], optional
        Refractive index of water.
    wavelength : float, optional
        Wavelength of the incident light in nanometers.
    discretize_Mie : bool, optional
        If True, discretizes input parameters for Mie calculations to enable
        caching.
    return_coefficients : bool, optional
        If True, returns the individual extinction coefficients for wet and
        dry aerosols instead of their ratio.
    return_all_optics : bool, optional
        If True, returns all optical properties calculated by Mie theory,
        not just extinction.

    Returns
    -------
    Union[float, Tuple[NDArray, NDArray]]
        By default, returns the ratio of wet to dry aerosol extinction.
        If `return_coefficients` is True, returns a tuple of NDArrays
        containing the extinction coefficients for wet and dry aerosols,
        respectively.
    """
    # Convert particle diameters to volumes for dry aerosol calculations
    volume_sizer = convert.length_to_volume(diameters, length_type='diameter')

    # Calculate volumes for solute and water in dry and wet conditions
    volume_dry = convert.kappa_volume_solute(
        volume_sizer, kappa, water_activity_sizer)
    volume_water_dry = convert.kappa_volume_water(
        volume_dry, kappa, water_activity_dry)
    volume_water_wet = convert.kappa_volume_water(
        volume_dry, kappa, water_activity_wet)

    # Determine effective refractive indices for dry and wet aerosols
    n_effective_dry = convert.effective_refractive_index(
        refractive_index_dry, water_refractive_index,
        volume_dry[-1],  # pyright: ignore[reportIndexIssue]
        volume_water_dry[-1]  # pyright: ignore[reportIndexIssue]
        )
    n_effective_wet = convert.effective_refractive_index(
        refractive_index_dry, water_refractive_index,
        volume_dry[-1],  # pyright: ignore[reportIndexIssue]
        volume_water_wet[-1]  # pyright: ignore[reportIndexIssue]
        )

    # Adjust diameters for wet and dry conditions and calculate optical
    # properties
    diameters_dry = convert.volume_to_length(
        volume_dry + volume_water_dry, length_type='diameter')
    diameters_wet = convert.volume_to_length(
        volume_dry + volume_water_wet, length_type='diameter')

    optics_dry = mie_size_distribution(
        m_sphere=n_effective_dry,
        wavelength=wavelength,
        diameter=diameters_dry,  # pyright: ignore[reportArgumentType]
        number_per_cm3=number_per_cm3,
        pms=True,
        extinction_only=not return_all_optics,
        discretize=discretize_Mie)

    optics_wet = mie_size_distribution(
        m_sphere=n_effective_wet,
        wavelength=wavelength,
        diameter=diameters_wet,  # pyright: ignore[reportArgumentType]
        number_per_cm3=number_per_cm3,
        pms=True,
        extinction_only=not return_all_optics,
        discretize=discretize_Mie)

    # Return either the extinction coefficients or their ratio based on user
    # choice
    if return_coefficients:
        return optics_wet, optics_dry
    return optics_wet / optics_dry  # pyright: ignore[reportOperatorIssue]


def fit_extinction_ratio_with_kappa(
    b_ext_dry: Union[float, np.float64],
    b_ext_wet: Union[float, np.float64],
    number_per_cm3: NDArray[np.float64],
    diameters: NDArray[np.float64],
    water_activity_sizer: NDArray[np.float64],
    water_activity_dry: NDArray[np.float64],
    water_activity_wet: NDArray[np.float64],
    refractive_index_dry: Union[complex, float] = 1.45,
    water_refractive_index: Union[complex, float] = 1.33,
    wavelength: float = 450,
    discretize_Mie: bool = True,
    kappa_bounds: Tuple[float, float] = (0, 1),
    kappa_tolerance: float = 1e-6,
    kappa_maxiter: int = 100,
) -> Union[float, np.float64]:
    """
    Fits the kappa parameter based on the measured extinction ratios of dry
    and wet aerosols, utilizing Mie theory to account for water uptake
    effects. This method optimizes kappa to minimize the difference between
    the calculated and observed extinction ratio of wet to dry aerosols.

    Parameters
    ----------
    b_ext_dry : Union[float, np.float64]
        The measured extinction of the dry aerosol.
    b_ext_wet : Union[float, np.float64]
        The measured extinction of the wet aerosol.
    number_per_cm3 : NDArray[np.float64]
        The number concentration of particles per cubic centimeter for each
        size bin.
    diameters : NDArray[np.float64]
        The diameters of particles in nanometers for each size bin.
    water_activity_sizer : NDArray[np.float64]
        The water activity corresponding to the aerosol size distribution.
    water_activity_dry : NDArray[np.float64]
        The water activity for the 'dry' aerosol condition.
    water_activity_wet : NDArray[np.float64]
        The water activity for the 'wet' aerosol condition.
    refractive_index_dry : Union[complex, float, np.float16], optional
        The refractive index of the dry aerosol particles.
    water_refractive_index : Union[complex, float], optional
        The refractive index of water.
    wavelength : float, optional
        The wavelength of incident light in nanometers.
    discretize_Mie : bool, optional
        If True, discretizes input parameters for Mie calculations to enable
        caching.
    kappa_bounds : Tuple[float, float], optional
        The bounds within which to fit the kappa parameter.
    kappa_tolerance : float, optional
        The tolerance level for the optimization of kappa.
    kappa_maxiter : int, optional
        The maximum number of iterations allowed in the optimization process.

    Returns
    -------
    Union[float, np.float64]
        The optimized kappa parameter that best fits the observed extinction
        ratios.
    """

    def objective_function(kappa_guess):
        """
        Objective function to minimize: the difference between the guessed
        extinction ratio (based on the current kappa guess) and the observed
        extinction ratio (wet/dry).
        """
        ratio_guess = extinction_ratio_wet_dry(
            kappa=kappa_guess,
            number_per_cm3=number_per_cm3,
            diameters=diameters,
            water_activity_sizer=water_activity_sizer,
            water_activity_dry=water_activity_dry,
            water_activity_wet=water_activity_wet,
            refractive_index_dry=refractive_index_dry,
            water_refractive_index=water_refractive_index,
            wavelength=wavelength,
            discretize_Mie=discretize_Mie,
            return_coefficients=False
        )
        # type check
        ratio_guess = ratio_guess \
            if isinstance(ratio_guess, float) \
            else float(ratio_guess)  # pyright: ignore
        return np.abs(ratio_guess - b_ext_wet / b_ext_dry)

    # Use fminbound to optimize the kappa value within the specified bounds
    # and tolerance
    kappa_opt, _, _, _ = fminbound(
        objective_function,
        x1=kappa_bounds[0],
        x2=kappa_bounds[1],
        xtol=kappa_tolerance,
        maxfun=kappa_maxiter,
        full_output=True,
        disp=0
    )
    return kappa_opt


@lru_cache(maxsize=100000)
def discretize_ScatteringFunction(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: Union[float, np.float64],
    minAngle: int = 0,
    maxAngle: int = 180,
    angularResolution: float = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Discretizes and caches the scattering function for a spherical particle
    with specified material properties and size. This function aims to optimize
    the performance of scattering calculations by caching results for
    frequently used parameters, reducing the need for repeated calculations.

    Parameters
    ----------
    m_sphere : Union[complex, float]
        The complex or real refractive index of the particle.
    wavelength : float
        The wavelength of the incident light in nanometers (nm).
    diameter : Union[float, np.float64]
        The diameter of the particle in nanometers (nm).
    minAngle : float, optional
        The minimum scattering angle in degrees to be considered in the
        calculation. Defaults to 0.
    maxAngle : float, optional
        The maximum scattering angle in degrees to be considered in the
        calculation. Defaults to 180.
    angularResolution : float, optional
        The resolution in degrees between calculated scattering angles.
        Defaults to 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    A tuple containing numpy arrays for the scattering function measurements:
        - measure: The scattering intensity as a function of angle.
        - SL: The scattering intensity for left-handed polarization.
        - SR: The scattering intensity for right-handed polarization.
        - SU: The unpolarized scattering intensity.
    """
    measure, SL, SR, SU = ps.ScatteringFunction(
        m=m_sphere,
        wavelength=wavelength,
        diameter=diameter,
        minAngle=minAngle,
        maxAngle=maxAngle,
        angularResolution=angularResolution
    )
    return measure, SL, SR, SU


def calculate_scattering_angles(
    z_position: Union[float, np.float64],
    integrate_sphere_diameter_cm: float,
    tube_diameter_cm: float
) -> Tuple[float, float]:
    """
    Calculates forward and backward scattering angles for a given position
    along the z-axis within the CAPS instrument geometry.

    Parameters
    ----------
    z_position : Union[float, np.float64]
        The position along the z-axis (in cm).
    integrate_sphere_diameter_cm : float
        The diameter of the integrating sphere (in cm).
    tube_diameter_cm : float
        The diameter of the sample tube (in cm).

    Returns
    -------
    Tuple[float, float]
        A tuple containing the forward (alpha) and backward (beta)
        scattering angles in radians.
    """
    sphere_radius_cm = integrate_sphere_diameter_cm / 2
    tube_radius_cm = tube_diameter_cm / 2

    # Calculate forward scattering angle alpha
    if z_position != sphere_radius_cm:
        alpha = np.arctan(tube_radius_cm / abs(sphere_radius_cm - z_position))
    else:
        alpha = np.pi / 2  # Edge case when directly at the edge of the sphere

    # Calculate backward scattering angle beta
    if z_position != -sphere_radius_cm:
        beta = np.arctan(tube_radius_cm / abs(sphere_radius_cm + z_position))
    else:
        beta = np.pi / 2  # Edge case when directly at the other edge of sphere

    return alpha, beta


def assign_scattering_thetas(
    alpha: float,
    beta: float,
    q_mie: float,
    z_position: Union[float, np.float64],
    integrate_sphere_diameter_cm: float
) -> Tuple[float, float, float]:
    """
    Assigns scattering angles and efficiencies based on the z-axis position
    within the CAPS instrument.

    Parameters
    ----------
    alpha : float
        Forward scattering angle in radians.
    beta : float
        Backward scattering angle in radians.
    q_mie : float
        The Mie scattering efficiency.
    z_position : Union[float, np.float64]
        The position along the z-axis (in cm).
    integrate_sphere_diameter_cm : float
        The diameter of the integrating sphere (in cm).

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the forward scattering angle (theta1), backward
        scattering angle (theta2), and the ideal scattering efficiency
        (qsca_ideal) for the given z-axis position.
    """
    sphere_radius_cm = integrate_sphere_diameter_cm / 2

    # Determine the location of z_position relative to the sphere's center
    # Outside the sphere
    if z_position < -sphere_radius_cm or z_position > sphere_radius_cm:
        qsca_ideal = 0
    else:  # Inside the sphere
        qsca_ideal = q_mie

    # Calculate the effective scattering angles based on z_position
    theta1 = alpha if z_position <= sphere_radius_cm else np.pi - alpha
    theta2 = np.pi - beta if z_position >= -sphere_radius_cm else beta

    return theta1, theta2, qsca_ideal


def get_truncated_scattering(
    scattering_unpolarized: np.ndarray,
    theta: np.ndarray,
    theta1: float,
    theta2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the truncated scattering intensity and corresponding angles based
    on the given truncation angles.

    Parameters
    ----------
    su : np.ndarray
        The scattering intensity for unpolarized light as a function of angle.
    theta : np.ndarray
        The array of angles corresponding to the scattering intensity
        measurements.
    theta1 : float
        The lower bound of the angle range for truncation in radians.
    theta2 : float
        The upper bound of the angle range for truncation in radians.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the truncated scattering intensity and the
        corresponding angles within the truncated range.
    """
    trunc_indices = np.where((theta >= theta1) & (theta <= theta2))
    scattering_unpolarized_trunc = scattering_unpolarized[trunc_indices]
    theta_trunc = theta[trunc_indices]

    return scattering_unpolarized_trunc, theta_trunc


@lru_cache(maxsize=100000)
def trunc_mono(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: float,
    fullOutput: bool = False,
    calTrunc: bool = False,
    discretize: bool = True,
) -> Union[float,
           Tuple[
               float, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray]]:
    """
    Calculates the single scattering albedo (SSA) correction due to truncation
    for monodisperse aerosol measurements using the CAPS-PM-SSA instrument. The
    correction accounts for the incomplete angular range of scattering
    measurements due to the instrument's geometry.

    Parameters
    ----------
    m_sphere : Union[complex, float]
        Complex or real refractive index of the aerosol.
    wavelength : float
        Wavelength of light in nanometers used in the CAPS instrument.
    diameter : Union[float, np.float64]
        Diameter of the monodisperse aerosol in nanometers.
    fullOutput : bool, optional
        If True, additional details about the calculation are returned,
        including z-axis values, angles of integration, and both truncated and
        ideal scattering efficiencies.
    calTrunc : bool, optional
        If True, applies a calibration factor to the truncation correction.
        Default is False.
    discretize : bool, optional
        If True, discretizes the input parameters for potentially improved
        stability/performance in scattering function calculations.

    Returns
    -------
    Union[float, Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray]]
    If fullOutput is False, returns only the truncation correction factor.
    If fullOutput is True, returns a tuple containing the truncation correction
    factor, z-axis positions, truncated scattering efficiency, ideal scattering
    efficiency, forward scattering angle, and backward scattering angle.
    """
    # Constants defining the geometry of the CAPS instrument
    diam_sphere = 10.0  # Diameter of integrating sphere in cm
    diam_tube = 1.0  # Diameter of sample tube in cm
    extra_length = 0.6  # Extra length considered outside the integrating
    # sphere in cm
    z1 = -0.5 * diam_sphere - extra_length  # Start of the z-axis integral
    # range
    z2 = 0.5 * diam_sphere + extra_length  # End of the z-axis integral range
    npos = 100  # Number of positions along the z-axis
    ang_res = 0.2  # Angular resolution in degrees

    # Calibration value for truncation at a reference diameter of 150nm
    trunc_calibration = 1.02245612148504

    # Calculate the size parameter
    size_param = (np.pi * diameter) / wavelength

    # Discretize parameters if enabled
    if discretize:
        m_sphere, wavelength, diameter = discretize_Mie_parameters(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter
        )

    # Choose the appropriate scattering function based on discretization
    # preference
    scattering_function = discretize_ScatteringFunction \
        if discretize else ps.ScatteringFunction
    theta, _, _, su = scattering_function(
        m_sphere,
        wavelength,
        diameter,
        minAngle=0,
        maxAngle=180,
        angularResolution=ang_res
    )

    # Integrate the Mie scattering efficiency over all angles
    q_mie = trapz((2 * su * np.sin(theta)) / size_param**2, theta)

    # Initialize arrays for z-axis positions, angles, and scattering
    # efficiencies
    z_axis = np.linspace(z1, z2, npos)
    theta1 = np.zeros(len(z_axis))
    theta2 = np.zeros(len(z_axis))
    qsca_trunc = np.zeros(len(z_axis))
    qsca_ideal = np.zeros(len(z_axis))

    # Calculate truncated and ideal scattering efficiencies across the z-axis
    for i, z in enumerate(z_axis):
        # Calculate forward and backward scattering angles based on geometry
        alpha, beta = calculate_scattering_angles(z, diam_sphere, diam_tube)

        # Assign scattering angles and efficiencies based on z-axis position
        theta1[i], theta2[i], qsca_ideal[i] = assign_scattering_thetas(
            alpha, beta, q_mie, z, diam_sphere)

        # Calculate truncated scattering efficiency for this z-axis position
        su_trunc, theta_trunc = get_truncated_scattering(
            su, theta, theta1[i], theta2[i])
        qsca_trunc[i] = trapz(
            (2 * su_trunc * np.sin(theta_trunc)) / size_param**2,
            theta_trunc)

    # Integrate scattering efficiencies over the z-axis to get total
    # efficiencies
    trunc = trapz(qsca_trunc, z_axis)
    ideal = trapz(qsca_ideal, z_axis)

    # Apply calibration factor to truncation correction if requested
    trunc_corr = (ideal / trunc) / \
        trunc_calibration if calTrunc else ideal / trunc

    return (trunc_corr, z_axis, qsca_trunc, qsca_ideal,
            theta1, theta2) if fullOutput else trunc_corr


@lru_cache(maxsize=100000)
def trunc_mono_legacy(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: Union[float, np.float64],
    fullOutput=False,
    calTrunc=False,
    discretize=True,
):  # sourcery skip
    """
    Compute the SSA correction due to truncation for the CAPS-PM-SSA for
    monodisperse aerosol. The truncation follows as the inverse of the SSA
    correction given. (i.e. ideal / truncation)

    Truncated and ideal Mie scattering efficiency several points within the
    geometry of the CAPS, and then integrated. The correction is derived from a
    ratio of the integrals (ideal case in the numerator).
    Informaiton about the z axis points used in the calculation, Mie scattering
    efficiencies, and angles used to compute truncated Mie scattering
    efficiencies can be output by setting fullOutput=True.

    Args
    ----------
    m: complex float
        Complex refrative index of the aerosol.
    diameter: float
        Diameter of monodisperse aerosol.
    wavelength: float, optional
        Wavelength of CAPS instrument. The default is 450.
    fullOutput: boolean, optional
        Outputs Scattering efficiencies, z-axis values, and angles of
        integration with correction.

    Returns
    -------
    trunc_corr: float
        Truncation for CAPS SSA measurement assuming monodisperse aerosol
    fullOutput=True also outputs
        trunc: float
            Truncated scattering efficiency. Integrated from theta1 to theta2
            across all z-axis values.
        ideal: float
            Non-truncated scattering efficiency. Integrated from 0-180 degrees
            z-axis values inside integrating sphere. Considered zero outside of
            limits of the integrating sphere.
        z_axis: array of floats
            z-axis positions of particles within cavity. Units are in cm.
        theta1: array of floats
            Forward scattering angle in radians of integration, truncated by
            opening on far side of cavity.
        theta2: array of floats
            Backward scattering angle in radians truncated by opening on near
            side of cavity.
    """
    # constants for the geometry of the CAPS
    diam_sphere = 10.0  # diameter of integrating tube in CAPS in cm
    diam_tube = 1.0  # diameter of tube in CAPS in cm
    extra_length = 0.6  # length outside both sides of integrating sphere
    # calculation is performed (cm)
    z1 = -0.5 * diam_sphere - extra_length  # lower limit of z-axis integral
    z2 = 0.5 * diam_sphere + extra_length  # upper limit of z-axis integral
    npos = 100
    angRes = 0.2  # Do not set angular resolution higher than 0.1

    trunc_calibration = 1.02245612148504  # truncation at calibration diameter
    # of 150nm trunc_calibration value seems to be dependent on angular
    # resolution of Scattering Function and positional resolution of z_axis.

    size_param = (np.pi * diameter) / wavelength  # size parameter

    # discretize parameters
    if discretize:
        # Discretize parameters for potentially improved stability/performance
        m_sphere, wavelength, diameter = discretize_Mie_parameters(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter
        )
    scattering_function = discretize_ScatteringFunction \
        if discretize else ps.ScatteringFunction
    theta, _, _, su = scattering_function(
        m_sphere,
        wavelength,
        diameter,
        minAngle=0,
        maxAngle=180,
        angularResolution=angRes
    )

    q_mie = trapz((2 * su * np.sin(theta)) / size_param**2, theta)

    z_axis = np.linspace(z1, z2, npos)
    theta1 = np.zeros(len(z_axis))
    theta2 = np.zeros(len(z_axis))
    qsca_trunc = np.zeros(len(z_axis))
    qsca_ideal = np.zeros(len(z_axis))
    i = 0

    for z in np.nditer(z_axis):
        if z != 0.5 * diam_sphere:
            alpha = np.arctan((0.5 * diam_tube) /
                              abs(0.5 * diam_sphere - z))  #
            # forward scattering angle
        else:
            pass
        if z != -0.5 * diam_sphere:
            beta = np.arctan((0.5 * diam_tube) / abs(-0.5 * diam_sphere - z))
            # back scattering angle

        # alpha and beta need to be adjusted based on section of cell we are in
        if z < -0.5 * diam_sphere:  # outside of sphere
            theta1[i] = alpha
            theta2[i] = beta
            qsca_ideal[i] = 0  # ideally light would not come from outside
        elif z == -0.5 * diam_sphere:
            theta1[i] = alpha
            theta2[i] = np.pi / 2
            qsca_ideal[i] = q_mie
        elif -0.5 * diam_sphere <= z <= 0.5 * diam_sphere:  # inside cavity
            theta1[i] = alpha
            theta2[i] = np.pi - beta
            qsca_ideal[i] = q_mie  # ideally full scattering efficiency measure
        elif z == 0.5 * diam_sphere:
            theta1[i] = np.pi / 2
            theta2[i] = np.pi - beta
            qsca_ideal[i] = q_mie  # ideally full scattering efficiency measure
        elif z > 0.5 * diam_sphere:
            theta1[i] = np.pi - alpha
            theta2[i] = np.pi - beta
            qsca_ideal[i] = 0  # ideally light would not get into cavity
            # from outside

        su_trunc = su[np.where(
            np.logical_and(theta >= theta1[i], theta <= theta2[i])
        )]
        theta_trunc = theta[np.where(
            np.logical_and(theta >= theta1[i], theta <= theta2[i])
        )]

        # calculate scattering efficiency
        qsca_trunc[i] = trapz(
            (2 * su_trunc * np.sin(theta_trunc)) / size_param**2,
            theta_trunc
        )
        i += 1
    trunc = trapz(qsca_trunc, z_axis)
    ideal = trapz(qsca_ideal, z_axis)

    if calTrunc:
        trunc_corr = ideal / trunc
    else:
        trunc_corr = (ideal / trunc) / trunc_calibration

    if fullOutput:
        return trunc_corr, z_axis, qsca_trunc, qsca_ideal, theta1, theta2
    else:
        return trunc_corr


def truncation_for_diameters(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter_sizes: NDArray[np.float64],
    discretize: bool = True
) -> NDArray[np.float64]:
    """
    Calculates the truncation correction for an array of particle diameters
    given a specific refractive index and wavelength. This function is
    particularly useful for aerosol optical property measurements where
    truncation effects due to instrument geometry need to be accounted for.

    Parameters
    ----------
    m_sphere : Union[complex, float]
        The complex or real refractive index of the particles.
    wavelength : float
        The wavelength of the incident light in nanometers (nm).
    diameter_sizes : NDArray[np.float64]
        An array of particle diameters in nanometers (nm) for which the
        truncation correction will be calculated.
    discretize : bool, optional
        A flag indicating whether to discretize the input parameters for
        potentially improved calculation performance. Default is True.

    Returns
    -------
    NDArray[np.float64]
        An array of truncation corrections corresponding to the input array of
        particle diameters.
    """
    truncation_array = np.zeros_like(diameter_sizes, dtype=np.float64)

    if discretize:
        m_sphere, wavelength, diameter_sizes = discretize_Mie_parameters(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter_sizes)

    # For each diameter, calculate the truncation correction
    for i, diameter in enumerate(diameter_sizes):
        truncation_array[i] = trunc_mono(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter,
            fullOutput=False,
            calTrunc=False,
            discretize=discretize
        )
    return truncation_array


def scattering_correction_for_distribution_measurements(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter_sizes: NDArray[np.float64],
    number_per_cm3: NDArray[np.float64],
    discretize: bool = True
) -> Union[float, np.float64]:
    """
    Calculates the correction factor for scattering measurements due to
    truncation effects in aerosol size distribution measurements. This
    correction factor is used to adjust the measured scattering
    coefficient, accounting for the limited angular range of the instrument.

    Parameters
    ----------
    m_sphere : Union[complex, float]
        The complex or real refractive index of the particles.
    wavelength : float
        The wavelength of the incident light in nanometers (nm).
    diameter_sizes : NDArray[np.float64]
        An array of particle diameters in nanometers (nm) corresponding to the
        size distribution.
    number_per_cm3 : NDArray[np.float64]
        An array of particle number concentrations (#/cm^3) for each diameter
        in the size distribution.
    discretize : bool, optional
        If True, the calculation will use discretized values for the
        refractive index, wavelength, and diameters to potentially improve
        computation performance. Default is True.

    Returns
    -------
    float
        The correction factor for scattering measurements. This factor is
        dimensionless and is used to correct the measured scattering
        coefficient for truncation effects, calculated as the ratio of
        the ideal (full angular range) to truncated scattering coefficient.

    Example
    -------
    b_sca_corrected = b_sca_measured * bsca_correction
    """
    # Calculate the truncation correction array for each diameter
    trunc_corr = truncation_for_diameters(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter_sizes=diameter_sizes,
        discretize=discretize
    )

    # Calculate the scattering coefficient with truncation effects
    _, b_sca_trunc, _, _, _, _ = mie_size_distribution(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter=diameter_sizes,
        number_per_cm3=number_per_cm3,
        discretize=discretize,
        truncation_calculation=True,
        truncation_b_sca_multiple=1 / trunc_corr
    )

    # Calculate the ideal (non-truncated) scattering coefficient
    _, b_sca_ideal, _, _, _, _ = mie_size_distribution(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter=diameter_sizes,
        number_per_cm3=number_per_cm3,
        discretize=discretize
    )

    # Return the correction factor as the ratio of ideal to truncated
    # scattering coefficients
    return b_sca_ideal / b_sca_trunc


def scattering_correction_for_humidified_measurements(
    kappa: Union[float, NDArray[np.float64]],
    number_per_cm3: NDArray[np.float64],
    diameter: NDArray[np.float64],
    water_activity_sizer: NDArray[np.float64],
    water_activity_sample: NDArray[np.float64],
    refractive_index_dry: Union[complex, float] = 1.45,
    water_refractive_index: Union[complex, float] = 1.33,
    wavelength: float = 450,
    discretize: bool = True
) -> NDArray[np.float64]:
    """
    Calculates the scattering correction for humidified aerosol measurements,
    accounting for water uptake by adjusting the aerosol's refractive index.
    This function requires the kappa values for the particles, which describe
    their hygroscopic growth.

    Parameters
    ----------
    kappa : Union[float, NDArray[np.float64]]
        Hygroscopicity parameter kappa, indicating the water uptake capability
        of the particles.
    number_per_cm3 : NDArray[np.float64]
        Number concentration of particles per cubic centimeter (#/cm³) for
        each size bin.
    diameter : NDArray[np.float64]
        Array of particle diameters in nanometers (nm).
    water_activity_sizer : NDArray[np.float64]
        Water activity (relative humidity/100) of the air sample used for
        sizing.
    water_activity_sample : NDArray[np.float64]
        Water activity (relative humidity/100) of the air sample in optical
        measurements.
    refractive_index_dry : Union[complex, float], optional
        Refractive index of the dry particles. Default is 1.45.
    water_refractive_index : Union[complex, float], optional
        Refractive index of water. Default is 1.33.
    wavelength : float, optional
        Wavelength of the incident light in nanometers (nm). Default is 450.
    discretize : bool, optional
        If True, calculation uses discretized values for refractive index,
        wavelength, and diameters to improve performance. Default is True.

    Returns
    -------
    NDArray[np.float64]
        A numpy array of scattering correction factors for each particle size
        in the distribution, to be applied to measured backscatter
        coefficients to account for truncation effects due to humidity.
    """
    # calculate the volume of the dry aerosol
    volume_sizer = convert.length_to_volume(diameter, length_type='diameter')
    volume_dry = convert.kappa_volume_solute(
        volume_sizer,
        kappa,
        water_activity_sizer
    )
    volume_water_sample = convert.kappa_volume_water(
        volume_dry,
        kappa,
        water_activity_sample
    )

    # calculate the effective refractive index of the dry+water aerosol
    m_sphere = convert.effective_refractive_index(
        refractive_index_dry,
        water_refractive_index,
        volume_water_sample[-1],
        volume_dry[-1]
    )
    # wet diameter sizes
    diameter_sizes = convert.volume_to_length(
        volume_dry + volume_water_sample,
        length_type='diameter'
    )
    # calculate the b_sca correction
    bsca_correction = scattering_correction_for_distribution_measurements(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter_sizes=diameter_sizes,
        number_per_cm3=number_per_cm3,
        discretize=discretize
    )
    return bsca_correction


def kappa_fitting_caps_data(
        extinction_dry: NDArray[np.float64],
        extinction_wet: NDArray[np.float64],
        number_per_cm3: NDArray[np.float64],
        diameter: NDArray[np.float64],
        water_activity_sizer: NDArray[np.float64],
        water_activity_sample_dry: NDArray[np.float64],
        water_activity_sample_wet: NDArray[np.float64],
        refractive_index_dry: complex | float = 1.45,
        water_refractive_index: complex | float = 1.33,
        wavelength: float = 450,
        discretize: bool = True,
) -> NDArray[np.float64]:
    """Fit he extinction ratio with kappa, looping over the time indexes
    in number_per_cm3

    Args
    ----------
    datalake : DataLake
        DataLake object

    Returns
    -------
    kappa : 2d array
        kappa 2d array, [kappa, lower, upper]
    bsca_truncation : array
        bsca truncation correction factor.
    """
    kappa_fit = np.zeros((extinction_wet, 3), dtype=np.float64)
    # Check for NaNs in 1D arrays
    nan_check_1d = np.isnan(extinction_dry) | np.isnan(extinction_wet) \
        | np.isnan(water_activity_sample_wet) | np.isnan(water_activity_sizer)\
            | np.isnan(water_activity_sample_dry)
    # Check for NaNs in 2D array along a specific
    nan_check_2d = np.isnan(number_per_cm3).any(axis=1)
    # Combine the checks
    skip_data = nan_check_1d | nan_check_2d

    # Loop over the time indexes
    for index, row_number_per_cm3 in enumerate(number_per_cm3):

        if skip_data[index]:
            kappa_fit[index, :] = np.nan
            continue  # skip if nan values

        kappa_fit[index, 0] = fit_extinction_ratio_with_kappa(
            b_ext_dry=extinction_dry[index],
            b_ext_wet=caps_wet[0],
            particle_counts=sizer_dn,
            diameters=sizer_diameter,
            water_activity_sizer=sizer_humidity / 100,
            water_activity_dry=caps_dry[1] / 100,
            water_activity_wet=np.mean(caps_wet[1:]) / 100,
            refractive_index_dry=refractive_index,
            water_refractive_index=1.33,
            wavelength=450,
            discretize_Mie=True,
            kappa_bounds=(0, 1),
            kappa_tolerance=1e-6,
            kappa_maxiter=100,
        )

        kappa_fit[i, 1] = fit_extinction_ratio_with_kappa(
            b_ext_dry=caps_dry[0],
            b_ext_wet=caps_wet[0],
            particle_counts=sizer_dn,
            diameters=sizer_diameter,
            water_activity_sizer=sizer_humidity / 100,
            water_activity_dry=caps_dry[1] / 100,
            water_activity_wet=np.mean(caps_wet[1:]) / 100,
            refractive_index_dry=refractive_index * 1.05,
            water_refractive_index=1.33,
            wavelength=450,
            discretize_Mie=True,
            kappa_bounds=(0, 1),
            kappa_tolerance=1e-6,
            kappa_maxiter=100,
        )

        kappa_fit[i, 2] = fit_extinction_ratio_with_kappa(
            b_ext_dry=caps_dry[0],
            b_ext_wet=caps_wet[0],
            particle_counts=sizer_dn,
            diameters=sizer_diameter,
            water_activity_sizer=sizer_humidity / 100,
            water_activity_dry=caps_dry[1] / 100,
            water_activity_wet=np.mean(caps_wet[1:]) / 100,
            refractive_index_dry=refractive_index * 0.95,
            water_refractive_index=1.33,
            wavelength=450,
            discretize_Mie=True,
            kappa_bounds=(0, 1),
            kappa_tolerance=1e-6,
            kappa_maxiter=100,
        )

    return kappa_fit


def truncation_precessing():
            if truncation_bsca:
                bsca_truncation_dry[i] = \
                    bsca_correction_for_humidified_measurements(
                        kappa=kappa_fit[i, 0],
                        particle_counts=sizer_dn,
                        diameters=sizer_diameter,
                        water_activity_sizer=sizer_humidity / 100,
                        water_activity_sample=caps_dry[1] / 100,
                        refractive_index_dry=refractive_index,
                        water_refractive_index=1.33,
                        wavelength=450,
                        discretize=True,
                        calibration_diameter=150,
                )
                bsca_truncation_wet[i] = \
                    bsca_correction_for_humidified_measurements(
                        kappa=kappa_fit[i, 0],
                        particle_counts=sizer_dn,
                        diameters=sizer_diameter,
                        water_activity_sizer=sizer_humidity / 100,
                        water_activity_sample=np.mean(caps_wet[1:]) / 100,
                        refractive_index_dry=refractive_index,
                        water_refractive_index=1.33,
                        wavelength=450,
                        discretize=True,
                        calibration_diameter=150,
                )
    return kappa_fit, bsca_truncation_dry, bsca_truncation_wet


def caps_processing(
        stream_size_distribution: Stream,
        stream_caps: Stream,
        truncation_bsca: bool = True,
        truncation_interval_sec: int = 600,
        truncation_interp: bool = True,
        refractive_index: float = 1.45,
        calibration_wet=1,
        calibration_dry=1,
        kappa_fixed: float = None,
):  # sourcery skip
    """loader.
    Function to process the CAPS data, and smps for kappa fitting, and then add
    it to the datalake. Also applies truncation corrections to the CAPS data.

    Args
    ----------
    datalake : object
        DataLake object to add the processed data to.
    truncation_bsca : bool, optional
        Whether to calculate truncation corrections for the bsca data.
        The default is True.
    truncation_interval_sec : int, optional
        The interval to calculate the truncation corrections over.
        The default is 600. This can take around 10 sec per data point.
    truncation_interp : bool, optional
        Whether to interpolate the truncation corrections to the caps data.
    refractive_index : float, optional
        The refractive index of the aerosol. The default is 1.45.
    calibration_wet : float, optional
        The calibration factor for the wet data. The default is 1.
    calibration_dry : float, optional
        The calibration factor for the dry data. The default is 1.

    Returns
    -------
    datalake : object
        DataLake object with the processed data added.
    """
    # calc kappa and add to datalake
    print('CAPS kappa_HGF fitting')

    if kappa_fixed is None:
        kappa_fit, _, _ = kappa_fitting_caps_data(
            datalake=datalake,
            truncation_bsca=False,
            refractive_index=refractive_index
        )
    else:
        kappa_len = len(
            datalake.datastreams['CAPS_data'].return_time(
                datetime64=False))
        kappa_fit = np.ones((kappa_len, 3)) * kappa_fixed

    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=kappa_fit.T,
        time_new=datalake.datastreams['CAPS_data'].return_time(
            datetime64=False),
        header_new=['kappa_fit', 'kappa_fit_lower', 'kappa_fit_upper'],
    )
    orignal_average = datalake.datastreams['CAPS_data'].average_base_sec
    orignal_average = datalake.datastreams['CAPS_data'].average_base_sec

    # calc truncation corrections and add to datalake
    print('CAPS truncation corrections')
    if truncation_bsca:
        datalake.reaverage_datastreams(
            truncation_interval_sec,
            stream_keys=['CAPS_data', 'smps_1D', 'smps_2D'],
        )
        # epoch_start=epoch_start,
        # epoch_end=epoch_end

        _, bsca_truncation_dry, bsca_truncation_wet = kappa_fitting_caps_data(
            datalake=datalake,
            truncation_bsca=truncation_bsca,
            refractive_index=refractive_index
        )

        if truncation_interp:
            interp_dry = interp1d(
                datalake.datastreams['CAPS_data'].return_time(
                    datetime64=False),
                bsca_truncation_dry,
                kind='linear',
                fill_value='extrapolate'
            )
            interp_wet = interp1d(
                datalake.datastreams['CAPS_data'].return_time(
                    datetime64=False),
                bsca_truncation_wet,
                kind='linear',
                fill_value='extrapolate'
            )

            time = datalake.datastreams['CAPS_data'].return_time(
                datetime64=False,
                raw=True
            )
            bsca_truncation_dry = interp_dry(time)
            bsca_truncation_wet = interp_wet(time)
        else:
            time = datalake.datastreams['CAPS_data'].return_time(
                datetime64=False)

        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=bsca_truncation_dry.T,
            time_new=time,
            header_new=['bsca_truncation_dry'],
        )
        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=bsca_truncation_wet.T,
            time_new=time,
            header_new=['bsca_truncation_wet'],
        )
    else:
        bsca_truncation_wet = np.array([1])
        bsca_truncation_dry = np.array([1])
        time = datalake.datastreams['CAPS_data'].return_time(
            datetime64=False,
            raw=True
        )

    # index for b_sca wet and dry
    index_dic = datalake.datastreams['CAPS_data'].return_header_dict()
    index_dic = datalake.datastreams['CAPS_data'].return_header_dict()

    # check if raw in dict
    if 'raw_b_sca_dry_CAPS_450nm[1/Mm]' in index_dic:
        pass
    else:
        # save raw data
        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=datalake.datastreams['CAPS_data'].data_stream[
                index_dic['b_sca_wet_CAPS_450nm[1/Mm]'], :],
            time_new=time,
            header_new=['raw_b_sca_wet_CAPS_450nm[1/Mm]'],
        )
        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=datalake.datastreams['CAPS_data'].data_stream[
                index_dic['b_sca_dry_CAPS_450nm[1/Mm]'], :],
            time_new=time,
            header_new=['raw_b_sca_dry_CAPS_450nm[1/Mm]'],
        )
        index_dic = datalake.datastreams['CAPS_data'].return_header_dict()

    datalake.datastreams['CAPS_data'].data_stream[
        index_dic['b_sca_wet_CAPS_450nm[1/Mm]'], :] = \
        datalake.datastreams['CAPS_data'].data_stream[
            index_dic['raw_b_sca_wet_CAPS_450nm[1/Mm]'], :] \
        * bsca_truncation_wet.T * calibration_wet

    datalake.datastreams['CAPS_data'].data_stream[
        index_dic['b_sca_dry_CAPS_450nm[1/Mm]'], :] = \
        datalake.datastreams['CAPS_data'].data_stream[
            index_dic['raw_b_sca_dry_CAPS_450nm[1/Mm]'], :] \
        * bsca_truncation_dry.T * calibration_dry

    datalake.datastreams['CAPS_data'].reaverage(
        reaverage_base_sec=orignal_average
    )  # updates the averages to the original value

    return datalake


def albedo_processing(
    datalake,
    keys: list = None
):
    """
    Calculates the albedo from the CAPS data and updates the datastream.

    Args
    ----------
    datalake : object
        DataLake object with the processed data added.

    Returns
    -------
    datalake : object
        DataLake object with the processed data added.
    """

    ssa_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['b_sca_wet_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['b_ext_wet_CAPS_450nm[1/Mm]'])[0]
    ssa_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['b_sca_dry_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['b_ext_dry_CAPS_450nm[1/Mm]'])[0]
    ssa_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['b_sca_wet_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['b_ext_wet_CAPS_450nm[1/Mm]'])[0]
    ssa_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['b_sca_dry_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['b_ext_dry_CAPS_450nm[1/Mm]'])[0]

    babs_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['b_ext_wet_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['b_sca_wet_CAPS_450nm[1/Mm]'])[0]
    babs_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['b_ext_dry_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['b_sca_dry_CAPS_450nm[1/Mm]'])[0]
    babs_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['b_ext_wet_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['b_sca_wet_CAPS_450nm[1/Mm]'])[0]
    babs_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['b_ext_dry_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['b_sca_dry_CAPS_450nm[1/Mm]'])[0]

    time = datalake.datastreams['CAPS_data'].return_time(datetime64=False)
    time = datalake.datastreams['CAPS_data'].return_time(datetime64=False)

    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=ssa_wet,
        time_new=time,
        header_new=['SSA_wet_CAPS_450nm[1/Mm]'],
    )
    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=ssa_dry,
        time_new=time,
        header_new=['SSA_dry_CAPS_450nm[1/Mm]'],
    )
    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=babs_wet,
        time_new=time,
        header_new=['b_abs_wet_CAPS_450nm[1/Mm]'],
    )
    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=babs_dry,
        time_new=time,
        header_new=['b_abs_dry_CAPS_450nm[1/Mm]'],
    )
    return datalake


def pass3_processing(
        datalake: object,
        babs_405_532_781=[1, 1, 1],
        bsca_405_532_781=[1, 1, 1],
):
    """
    Processing PASS3 data applying the calibration factors
    TODO: add the background correction

    Args
    ----------
    datalake : object
        DataLake object to add the processed data to.
    babs_405_532_781 : list, optional
        Calibration factors for the absorption channels. The default is [1,1,1]
    bsca_405_532_781 : list, optional
        Calibration factors for the scattering channels. The default is [1,1,1]

    Returns
    -------
    datalake : object
        DataLake object with the processed data added.
    """
    # index for b_sca wet and dry
    index_dic = datalake.datastreams['pass3'].return_header_dict()
    time = datalake.datastreams['pass3'].return_time(
        datetime64=False,
        raw=True
    )
    babs_list = ['b_abs405nm[1/Mm]', 'b_abs532nm[1/Mm]', 'b_abs781nm[1/Mm]']
    bsca_list = ['b_sca405nm[1/Mm]', 'b_sca532nm[1/Mm]', 'b_sca781nm[1/Mm]']

    if 'raw_b_abs405nm[1/Mm]' not in index_dic:
        print('Copy raw babs Pass-3')
        for babs in babs_list:
            raw_name = f'raw_{babs}'

            datalake.datastreams['pass3'].add_processed_data(
                data_new=datalake.datastreams['pass3'].data_stream[
                    index_dic[babs], :],
                time_new=time,
                header_new=[raw_name],
            )
    if 'raw_b_sca405nm[1/Mm]' not in index_dic:
        print('Copy raw bsca Pass-3')
        for bsca in bsca_list:
            raw_name = f'raw_{bsca}'

            datalake.datastreams['pass3'].add_processed_data(
                data_new=datalake.datastreams['pass3'].data_stream[
                    index_dic[bsca], :],
                time_new=time,
                header_new=[raw_name],
            )

    index_dic = datalake.datastreams['pass3'].return_header_dict()

    # calibration loop babs.
    print('Calibrated raw Pass-3')
    for i, babs in enumerate(babs_list):
        raw_name = f'raw_{babs}'
        datalake.datastreams['pass3'].data_stream[index_dic[babs], :] = \
            datalake.datastreams['pass3'].data_stream[index_dic[raw_name], :] \
            * babs_405_532_781[i]
    # calibration loop bsca
    for i, bsca in enumerate(bsca_list):
        raw_name = f'raw_{bsca}'
        datalake.datastreams['pass3'].data_stream[index_dic[bsca], :] = \
            datalake.datastreams['pass3'].data_stream[index_dic[raw_name], :] \
            * bsca_405_532_781[i]

    return datalake
