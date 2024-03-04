"""Scattering Truncation Correction Functions"""
# pylint: disable=too-many-arguments, too-many-locals

from typing import Union, Tuple
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray

from tqdm import tqdm
import PyMieScatt as ps
from scipy.integrate import trapz
from particula.util import convert
from particula.data.process import mie_angular, mie_bulk


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
    full_output: bool = False,
    calibrated_trunc: bool = True,
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
    full_output : bool, optional
        If True, additional details about the calculation are returned,
        including z-axis values, angles of integration, and both truncated and
        ideal scattering efficiencies.
    calibrated_trunc : bool, optional
        If True, applies a numberical calibration factor to the truncation
        correction, so 150 is 1. Default is True.
    discretize : bool, optional
        If True, discretizes the input parameters for potentially improved
        stability/performance in scattering function calculations. Can not be
        done for full_output=True

    Returns
    -------
    Union[float, Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray]]
    If fullOutput is False, returns only the truncation correction factor.
    If fullOutput is True, returns a tuple containing the truncation correction
    factor, z-axis positions, truncated scattering efficiency, ideal scattering
    efficiency, forward scattering angle, and backward scattering angle.
    """
    if full_output:
        discretize = False  # can not have full output discretized due to hash

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
        m_sphere, wavelength, diameter = mie_bulk.discretize_mie_parameters(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter
        )

    # Choose the appropriate scattering function based on discretization
    # preference
    if discretize:
        theta, _, _, su = mie_angular.discretize_scattering_angles(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter,
            min_angle=0,
            max_angle=180,
            angular_resolution=ang_res
        )
    else:
        theta, _, _, su = ps.ScatteringFunction(
            m=m_sphere,
            wavelength=wavelength,
            diameter=diameter,
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
        alpha, beta = mie_angular.calculate_scattering_angles(
            z, diam_sphere, diam_tube)

        # Assign scattering angles and efficiencies based on z-axis position
        theta1[i], theta2[i], qsca_ideal[i] = \
            mie_angular.assign_scattering_thetas(
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
        trunc_calibration if calibrated_trunc else ideal / trunc

    return (trunc_corr, z_axis, trunc, ideal,
            theta1, theta2) if full_output else trunc_corr


def truncation_for_diameters(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter_sizes: NDArray[np.float64],
    discretize: bool = True,
    calibrated_trunc: bool = True
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
    calibrated_trunc : bool, optional
        If True, applies a numberical calibration factor to the truncation
        correction, so 150 is 1. Default is True.

    Returns
    -------
    NDArray[np.float64]
        An array of truncation corrections corresponding to the input array of
        particle diameters.
    """
    truncation_array = np.zeros_like(diameter_sizes, dtype=np.float64)

    if discretize:
        m_sphere, wavelength, diameter_sizes = \
            mie_bulk.discretize_mie_parameters(
                m_sphere=m_sphere,
                wavelength=wavelength,
                diameter=diameter_sizes)

    # For each diameter, calculate the truncation correction
    for i, diameter in enumerate(diameter_sizes):
        truncation_array[i] = trunc_mono(
            m_sphere=m_sphere,
            wavelength=wavelength,
            diameter=diameter,
            full_output=False,
            calibrated_trunc=calibrated_trunc,
            discretize=discretize
        )
    return truncation_array


def correction_for_distribution(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter_sizes: NDArray[np.float64],
    number_per_cm3: NDArray[np.float64],
    discretize: bool = True,
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
    _, b_sca_trunc, _, _, _, _, _ = mie_bulk.mie_size_distribution(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter=diameter_sizes,
        number_per_cm3=number_per_cm3,
        discretize=discretize,
        truncation_calculation=True,
        truncation_b_sca_multiple=1 / trunc_corr
    )

    # Calculate the ideal (non-truncated) scattering coefficient
    _, b_sca_ideal, _, _, _, _, _ = mie_bulk.mie_size_distribution(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter=diameter_sizes,
        number_per_cm3=number_per_cm3,
        discretize=discretize
    )

    # Return the correction factor as the ratio of ideal to truncated
    # scattering coefficients
    return b_sca_ideal / b_sca_trunc


def correction_for_humidified(
    kappa: Union[float, np.float64],
    number_per_cm3: NDArray[np.float64],
    diameter: NDArray[np.float64],
    water_activity_sizer: np.float64,
    water_activity_sample: np.float64,
    refractive_index_dry: Union[complex, float] = 1.45,
    water_refractive_index: Union[complex, float] = 1.33,
    wavelength: float = 450,
    discretize: bool = True
) -> np.float64:
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
    np.float64
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
    bsca_correction = correction_for_distribution(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter_sizes=diameter_sizes,
        number_per_cm3=number_per_cm3,
        discretize=discretize,
    )
    return bsca_correction


def correction_for_humidified_looped(
    kappa: NDArray[np.float64],
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
    Corrects scattering measurements for aerosol particles to account for
    truncation errors in CAPS instrument. This correction is vital for
    accurate representation of particle scattering properties under different
    humidity conditions. The function iterates over time-indexed measurements,
    calculating corrections based on input parameters reflecting the particles'
    physical and chemical characteristics.

    Parameters
    ----------
    kappa : NDArray[np.float64]
        Hygroscopicity parameter array for the aerosol particles, indicating
        water uptake ability.
    number_per_cm3 : NDArray[np.float64]
        Time-indexed number concentration of particles in #/cm³ for each size.
    diameter : NDArray[np.float64]
        Particle diameters, crucial for calculating scattering effects.
    water_activity_sizer : NDArray[np.float64]
        Water activity measured by the sizing instrument, indicating relative
        humidity.
    water_activity_sample : NDArray[np.float64]
        Sample water activity, corresponding to the ambient conditions during
        measurement.
    refractive_index_dry : Union[complex, float], optional
        Refractive index of the dry particles, affecting their scattering
        behavior. Default is 1.45.
    water_refractive_index : Union[complex, float], optional
        Refractive index of water, important for calculations involving
        humidified conditions. Default is 1.33.
    wavelength : float, optional
        Wavelength of the incident light in nanometers, which influences
        scattering intensity. Default is 450.
    discretize : bool, optional
        If set to True, performs discretized calculations for potentially
        improved computational performance. Default is True.

    Returns
    -------
    NDArray[np.float64]
    An array of corrected scattering multipliers for each time index,
    accounting for aerosol particle size, composition, and environmental
    conditions.

    The correction process includes data nan checks for missing values,
    ensuring robust and reliable correction outcomes.
    """
    # Initialize an array to store the correction factors
    correction_multiple = np.zeros((len(kappa)), dtype=np.float64)

    # Determine indices with missing input data to exclude from correction
    skip_data = np.isnan(kappa) | \
        np.isnan(water_activity_sample) | \
        np.isnan(water_activity_sizer) | \
        np.isnan(number_per_cm3).any(axis=1)

    # Iterate over each time index to apply the scattering correction
    for index, row_number_per_cm3 in tqdm(enumerate(number_per_cm3),
                                          'Processing Scattering Correction'):
        if skip_data[index]:
            # Assign NaN for indices with incomplete data
            correction_multiple[index] = np.nan
            continue
        # Correct scattering for the current time index using the specified
        # parameters
        correction_multiple[index] = \
            correction_for_humidified(
                kappa=kappa[index],
                number_per_cm3=row_number_per_cm3,
                diameter=diameter,
                water_activity_sizer=water_activity_sizer[index],
                water_activity_sample=water_activity_sample[index],
                refractive_index_dry=refractive_index_dry,
                water_refractive_index=water_refractive_index,
                wavelength=wavelength,
                discretize=discretize
        )
    return correction_multiple
