"""Functions for processing optical data."""
# linting disabled until reformatting of this file
# pylint: disable=all
# pytype: skip-file
# flake8: noqa

from typing import Optional, Union
import numpy as np
from scipy.interpolate import interp1d
from particula.data.stream import Stream

import numpy as np
import PyMieScatt as ps
from scipy.integrate import trapz
from functools import lru_cache
from tqdm import tqdm
from scipy.optimize import fminbound
from particula.util import convert


from typing import Tuple, Union
from functools import lru_cache
import PyMieScatt as ps

@lru_cache(maxsize=100000)
def discretize_AutoMieQ(
    m: Union[complex, float],
    wavelength: float,
    dp: float,
    nMedium: Union[complex, float] = 1.0,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Computes Mie coefficients for a spherical particle based on its material
    properties, size, and the properties of the surrounding medium.

    This function leverages the PyMieScatt library to calculate the extinction
    (Q_ext), scattering (Q_sca), absorption (Q_abs) efficiencies, the
    asymmetry factor (g), radiation pressure efficiency (Q_pr), backscatter
    efficiency (Q_back), and the ratio of backscatter to extinction efficiency
    (Q_ratio) for a single sphere under specified conditions.

    This function is optimized with an LRU (Least Recently Used) cache to
    enhance performance by storing up to 100,000 recent calls. The cache
    memorizes the results of expensive function calls and returns the cached
    result when the same inputs occur again, reducing the need to recompute
    these values.

    Args
    ----------
    m : The complex refractive index of the sphere. For non-absorbing material
        a real number can be provided.
    wavelength : The wavelength of the incident light in nanometers (nm).
    dp : The diameter of the sphere in nanometers (nm).
    nMedium : The refractive index of the surrounding medium.
        Defaults to 1.0, corresponding to vacuum.

    Returns
    -------
    Tuple[float, float, float, float, float, float, float]
        A tuple containing the calculated Mie efficiencies and parameters:
        Q_ext (extinction efficiency), Q_sca (scattering efficiency),
        Q_abs (absorption efficiency), g (asymmetry factor),
        Q_pr (radiation pressure efficiency), Q_back (backscatter efficiency),
        and Q_ratio (the ratio of backscatter to extinction efficiency).
    """
    return ps.AutoMieQ(m=m,
                       wavelength=wavelength,
                       diameter=dp,
                       nMedium=nMedium)


def discretize_parameters(
    m: Union[complex, float],
    wavelength: float,
    dp: np.ndarray
) -> Tuple[Union[complex, float], float, np.ndarray]:
    """
    Discretizes material refractive index, wavelength, and particle diameters.

    Parameters
    ----------
    m : Union[complex, float]
        The complex or real refractive index of the particles.
    wavelength : float
        The incident light wavelength in nanometers.
    dp : np.ndarray
        An array of particle diameters in nanometers.

    Returns
    -------
    Tuple[Union[complex, float], float, np.ndarray]
        The discretized values of m, wavelength, and dp.
    """
    m_real = convert.round_arbitrary(
        values=np.real(m),
        base=0.001,
        mode='round')
    m_imag = convert.round_arbitrary(
        values=np.imag(m),
        base=0.001,
        mode='round')
    # recombine the discretized real and imaginary parts
    m_discretized = m_real + 1j * m_imag if m_imag != 0 else m_real

    # discretize the wavelength assumes nm
    wavelength_discretized = convert.round_arbitrary(
        values=wavelength,
        base=1,
        mode='round')

    # discretize the particle diameters assumes nm
    dp_discretized = convert.round_arbitrary(
        values=dp,
        base=5,
        mode='round',
        nonzero_edge=True)

    return m_discretized, wavelength_discretized, dp_discretized


def compute_bulk_optics(
    q_ext: np.ndarray,
    q_sca: np.ndarray,
    q_back: np.ndarray,
    q_ratio: np.ndarray,
    g: np.ndarray,
    aSDn: np.ndarray,
    extinction_only: np.ndarray,
    pms: np.ndarray,
    dp: np.ndarray
):
    """
    Computes bulk optical properties from individual efficiency factors and
    size distribution.

    Parameters
    ----------
    q_ext, q_sca, q_abs, q_pr, q_back, q_ratio, g : np.ndarray
        Arrays of efficiency factors and the asymmetry factor.
    aSDn : np.ndarray
        Number size distribution.
    extinction_only : bool
        Flag to compute only the extinction coefficient.
    pms : bool
        Flag indicating if probability mass distribution, where sum of
        all bins is total number of particles, is used.
    dp : np.ndarray
        An array of particle diameters in nanometers.

    Returns
    -------
    Tuple containing bulk optical properties.
    """
    if pms:
        Bext = np.sum(q_ext * aSDn)
        if extinction_only:
            return Bext
        Bsca = np.sum(q_sca * aSDn)
        Babs = Bext - Bsca
        Bback = np.sum(q_back * aSDn)
        Bratio = np.sum(q_ratio * aSDn)
        bigG = np.sum(g * q_sca * aSDn) / Bsca if Bsca != 0 else 0
        Bpr = Bext - bigG * Bsca
    else:
        Bext = np.trapz(q_ext * aSDn, dp)
        if extinction_only:
            return Bext
        Bsca = np.trapz(q_sca * aSDn, dp)
        Babs = Bext - Bsca
        Bback = np.trapz(q_back * aSDn, dp)
        Bratio = np.trapz(q_ratio * aSDn, dp)
        bigG = np.trapz(g * q_sca * aSDn, dp) / Bsca if Bsca != 0 else 0
        Bpr = Bext - bigG * Bsca

    return Bext, Bsca, Babs, Bpr, Bback, Bratio, bigG


def format_results(Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio, asDict: bool) -> Union[Dict[str, float], Tuple[float, ...]]:
    """
    Formats the results of the Mie scattering calculations.

    Parameters
    ----------
    Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio : float
        Bulk optical properties.
    asDict : bool
        Determines the format of the results.

    Returns
    -------
    Either a dictionary or a tuple of the computed bulk optical properties.
    """
    if asDict:
        return {
            'Bext': Bext,
            'Bsca': Bsca,
            'Babs': Babs,
            'G': bigG,
            'Bpr': Bpr,
            'Bback': Bback,
            'Bratio': Bratio
        }
    else:
        return Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio


def Mie_SD(
    m: Union[complex, float],
    wavelength: float,
    dp: np.ndarray,
    ndp: np.ndarray,
    nMedium: Union[complex, float] = 1.0,
    SMPS: bool = True,
    asDict: bool = False,
    extinction_only: bool = False,
    discretize: bool = False,
    truncation_calculation: bool = False,
    truncation_Bsca_multiple: Optional[float] = None
) -> Union[float, Dict[str, float], Tuple[float, ...]]:
    """
    Calculates Mie scattering parameters for a size distribution of spherical particles.

    This function computes optical properties such as extinction, scattering, absorption
    coefficients, asymmetry factor, backscatter efficiency, and their ratios for a given
    size distribution of spherical particles. It supports various modes of calculation,
    including discretization of input parameters and optional truncation of the scattering
    efficiency.

    Parameters
    ----------
    m : Union[complex, float]
        The complex refractive index of the particles. Real values can be used for
        non-absorbing materials.
    wavelength : float
        The wavelength of the incident light in nanometers (nm).
    dp : np.ndarray
        An array of particle diameters in nanometers (nm).
    ndp : np.ndarray
        The number distribution of particles per cubic centimeter (#/cm^3).
    nMedium : Union[complex, float], optional
        The refractive index of the medium. Defaults to 1.0 (air or vacuum).
    SMPS : bool, optional
        Specifies if the size distribution follows the SMPS (Scanning Mobility Particle Sizer) convention.
        Defaults to True.
    asDict : bool, optional
        If True, results are returned as a dictionary. Otherwise, as a tuple. Defaults to False.
    extinction_only : bool, optional
        If True, only the extinction coefficient is calculated and returned. Defaults to False.
    discretize : bool, optional
        If True, input parameters (m, wavelength, dp) are discretized for computation. Defaults to False.
    truncation_calculation : bool, optional
        Enables truncation of the scattering efficiency based on a multiple of the backscattering coefficient.
        Defaults to False.
    truncation_Bsca_multiple : Optional[float], optional
        The multiple of the backscattering coefficient used for truncating the scattering efficiency.
        Required if `truncation_calculation` is True.

    Returns
    -------
    Union[float, Dict[str, float], Tuple[float, ...]]
        Depending on the parameters `asDict` and `extinction_only`, the function can return:
        - A single float (extinction coefficient) if `extinction_only` is True.
        - A dictionary of computed optical properties if `asDict` is True.
        - A tuple of computed optical properties.

    Raises
    ------
    ValueError
        If `truncation_calculation` is True but `truncation_Bsca_multiple` is not specified.

    Notes
    -----
    The discretization and truncation features allow for finer control over the calculation
    precision and the handling of outliers in size distribution, respectively. The use of SMPS
    convention influences the integration method for computing bulk properties from the size
    distribution.
    """
    # Adjust input parameters for medium's refractive index
    m /= nMedium
    wavelength /= nMedium

    # Ensure inputs are numpy arrays for vectorized operations
    dp, ndp = map(lambda x: convert.coerce_type(x, np.ndarray), (dp, ndp))

    # Initialize arrays for Mie efficiencies and asymmetry factor
    Q_ext, Q_sca, Q_abs, Q_pr, Q_back, Q_ratio, g = (np.zeros(dp.size) for _ in range(7))

    # Calculate area size distribution normalized to inverse megameters
    aSDn = np.pi * (dp / 2)**2 * ndp * 1e-6

    # Perform computations, optionally with discretization
    if discretize:
        # Discretize parameters for potentially improved stability/performance
        m_discretized, wavelength_discretized, dp_discretized = discretize_parameters(m, wavelength, dp)
        for i in range(dp.size):
            Q_ext[i], Q_sca[i], Q_abs[i], g[i], Q_pr[i], Q_back[i], Q_ratio[i] = discretize_AutoMieQ(
                m_discretized, wavelength_discretized, dp_discretized[i], nMedium)
    else:
        for i in range(dp.size):
            Q_ext[i], Q_sca[i], Q_abs[i], g[i], Q_pr[i], Q_back[i], Q_ratio[i] = ps.AutoMieQ(
                m, wavelength, dp[i], nMedium)

    # Apply optional truncation to scattering efficiency
    if truncation_calculation:
        if truncation_Bsca_multiple is None:
            raise ValueError("truncation_Bsca_multiple must be specified for truncation calculation.")
        Q_sca *= truncation_Bsca_multiple

    # Compute bulk optical properties based on size distribution and selected mode
    Bext, Bsca, Babs, Bpr, Bback, Bratio, bigG = compute_bulk_properties(
        Q_ext, Q_sca, Q_abs, Q_pr, Q_back, Q_ratio, g, aSDn, extinction_only, SMPS, dp)

    # Return results based on requested format
    if extinction_only:
        return Bext
    return format_results(Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio, asDict)


def Mie_SD(
        m: Union[complex, float],
        wavelength: float,
        dp: np.ndarray,
        ndp: np.ndarray,
        nMedium: Union[complex, float]=1.0,
        SMPS: bool=True,
        asDict: bool=False,
        extinction_only: bool=False,
        discretize: bool=False,
        truncation_calculation: bool=False,
        truncation_Bsca_multiple=None):  # sourcery skip
    """Return the extinction coefficient only if set to true

    Args
    ----------
    m : Complex refractive index of the sphere
    wavelength :      Wavelength of the incident light in nm
    dp : Diameter of the sphere in nm
    ndp :   Number distribution of the sphere #/cm^3
    nMedium : Refractive index of the medium, by default 1.0
    SMPS :True if the size distribution is SMPS, by default True
    interpolate :     True if the size distribution is interpolated, by default False
    asDict : if the output is a dictionary, by default False
    extinction_only : if only the extinction coefficient is returned, by default False
    discretize :    True if the size distribution is discretized, by default False
    truncation_calculation : if the size distribution is truncated, by default False
    truncation_Bsca_multiple :  multiple of the backscattering coefficient to truncate the Qsca
        coefficent by default None. Qsca_trunc = Qsca*truncation_Bsca_multiple

    Returns
    -------
    B_ext, B_sca, B_abs, g, B_pr, B_back, B_ratio
        Returns as a comma separated list or dictionary if asDict is True
    extinction_only : True
        Returns only the extinction coefficient 1/Mm
    """
    #  http://pymiescatt.readthedocs.io/en/latest/forward.html#Mie_SD
    nMedium = nMedium.real
    m /= nMedium
    wavelength /= nMedium
    dp = convert.coerce_type(dp, np.ndarray)
    ndp = convert.coerce_type(ndp, np.ndarray)
    _length = np.size(dp)
    Q_ext = np.zeros(_length)
    Q_sca = np.zeros(_length)
    Q_abs = np.zeros(_length)
    Q_pr = np.zeros(_length)
    Q_back = np.zeros(_length)
    Q_ratio = np.zeros(_length)
    g = np.zeros(_length)

    # scaling of 1e-6 to cast in units of inverse megameters - see docs
    aSDn = np.pi * ((dp / 2)**2) * ndp * (1e-6)

    if discretize:
        m_real = convert.round_arbitrary(np.real(m), base=0.001, mode='round')
        m_imag = convert.round_arbitrary(np.imag(m), base=0.001, mode='round')
        if m_imag == 0:
            m_discretized = m_real
        else:
            m_discretized = m_real + 1j * m_imag
        wavelength_discretized = convert.round_arbitrary(
            wavelength,
            base=1,
            mode='round'
        )
        dp_discretized = convert.round_arbitrary(
            dp,
            base=5,
            mode='round',
            nonzero_edge=True
        )
        for i in range(_length):
            Q_ext[i], Q_sca[i], Q_abs[i], g[i], Q_pr[i], Q_back[i], Q_ratio[i]\
                = discretize_AutoMieQ(
                    m_discretized,
                    wavelength_discretized,
                    dp_discretized[i],
                    nMedium
            )
    else:
        for i in range(_length):
            Q_ext[i], Q_sca[i], Q_abs[i], g[i], Q_pr[i], Q_back[i], Q_ratio[i]\
                = ps.AutoMieQ(m, wavelength, dp[i], nMedium)

    # apply truncation of bsca if requested
    if truncation_calculation:
        if truncation_Bsca_multiple is None:
            raise ValueError('truncation_Bsca_multiple must be specified')
        else:
            Q_sca = Q_sca * truncation_Bsca_multiple

    if SMPS:
        if extinction_only:
            Bext = np.sum(Q_ext * aSDn)
        else:
            Bext = np.sum(Q_ext * aSDn)
            Bsca = np.sum(Q_sca * aSDn)
            Babs = Bext - Bsca
            Bback = np.sum(Q_back * aSDn)
            Bratio = np.sum(Q_ratio * aSDn)
            bigG = np.sum(g * Q_sca * aSDn) / np.sum(Q_sca * aSDn)
            Bpr = Bext - bigG * Bsca
    else:
        if extinction_only:
            Bext = trapz(Q_ext * aSDn, dp)
        else:
            Bext = trapz(Q_ext * aSDn, dp)
            Bsca = trapz(Q_sca * aSDn, dp)
            Babs = Bext - Bsca
            Bback = trapz(Q_back * aSDn, dp)
            Bratio = trapz(Q_ratio * aSDn, dp)
            bigG = trapz(g * Q_sca * aSDn, dp) / trapz(Q_sca * aSDn, dp)
            Bpr = Bext - bigG * Bsca

    if extinction_only:
        return Bext
    else:
        if asDict:
            return dict(Bext=Bext, Bsca=Bsca, Babs=Babs, G=bigG,
                        Bpr=Bpr, Bback=Bback, Bratio=Bratio)
        else:
            return Bext, Bsca, Babs, bigG, Bpr, Bback, Bratio


def extinction_ratio_wet_dry(
        kappa,
        particle_counts,
        diameters,
        water_activity_sizer,
        water_activity_dry,
        water_activity_wet,
        refractive_index_dry=1.45,
        water_refractive_index=1.33,
        wavelength=450,
        discretize_Mie=True,
        return_coefficients=False,
        return_all_optics=False,
):  # sourcery skip
    """
    Calculate the extinction ratio of a dry aerosol to a wet aerosol.
    Using a kappa for water uptake. Uses Mie theory to calculate the extinction

    Args
    ----------
    kappa : float
        kappa parameter
    particle_counts : array_like
        particle counts for each diameter bin
    diameters : array_like
        diameter of each bin
    water_activity_sizer : float
        water activity of the size distribution
    water_activity_dry : float
        water activity of the 'dry' aerosol extinction
    water_activity_wet : float
        water activity of the 'wet' aerosol extinction
    refractive_index_dry : float optional
        refractive index of the dry aerosol
    water_refractive_index : float optional
        refractive index of water
    wavelength : float optional
        wavelength of light in nm
    discretize_Mie : bool optional
        discretize the Mie calculation so it can be cached
    return_coefficients : bool optional
        return the extinction of the wet and dry aerosol
    return_all_optics : bool optional
        return all the optics of the wet and dry aerosol

    Returns
    -------
    extinction_ratio : float
        extinction ratio of wet / dry aerosol
    """

    # calculate the volume of the dry aerosol
    volume_sizer = convert.length_to_volume(diameters, length_type='diameter')

    volume_dry = convert.kappa_volume_solute(
        volume_sizer,
        kappa,
        water_activity_sizer
    )
    volume_water_dry = convert.kappa_volume_water(
        volume_dry,
        kappa,
        water_activity_dry
    )
    volume_water_wet = convert.kappa_volume_water(
        volume_dry,
        kappa,
        water_activity_wet
    )

    # calculate the effective refractive index of the dry and wet aerosol
    n_effective_dry = convert.effective_refractive_index(
        refractive_index_dry,
        water_refractive_index,
        volume_dry[-1],
        volume_water_dry[-1]
    )
    n_effective_wet = convert.effective_refractive_index(
        refractive_index_dry,
        water_refractive_index,
        volume_dry[-1],
        volume_water_wet[-1]
    )

    # calculate the extinction for the dry and wet aerosol
    optics_dry = Mie_SD(
        n_effective_dry,
        wavelength,
        dp=convert.volume_to_length(
            volume_dry + volume_water_dry,
            length_type='diameter'),
        ndp=particle_counts,
        SMPS=True,
        extinction_only=not (return_all_optics),
        discretize=discretize_Mie
    )
    optics_wet = Mie_SD(
        n_effective_wet,
        wavelength,
        dp=convert.volume_to_length(
            volume_dry + volume_water_wet,
            length_type='diameter'),
        ndp=particle_counts,
        SMPS=True,
        extinction_only=not (return_all_optics),
        discretize=discretize_Mie
    )

    if return_coefficients:
        return optics_wet, optics_dry
    else:
        return optics_wet / optics_dry


def fit_extinction_ratio_with_kappa(
        Bext_dry,
        Bext_wet,
        particle_counts,
        diameters,
        water_activity_sizer,
        water_activity_dry,
        water_activity_wet,
        refractive_index_dry=1.45,
        water_refractive_index=1.33,
        wavelength=450,
        discretize_Mie=True,
        kappa_bounds=(0, 1),
        kappa_tolerance=1e-6,
        kappa_maxiter=100,
):  # sourcery skip
    """
    Fit the extinction ratio of a dry aerosol to a wet aerosol.
    Using a kappa for water uptake. Uses Mie theory to calculate the extinction
    The defaults are for the 450 nm wavelength.

    Args
    ----------
    Bext_dry : float
        measured extinction of the dry aerosol
    Bext_wet : float
        measured extinction of the wet aerosol
    particle_counts : array_like
        particle counts for each diameter bin
    diameters : array_like
        diameter of each bin
    water_activity_sizer : float
        water activity of the size distribution
    water_activity_dry : float
        water activity of the 'dry' aerosol extinction
    water_activity_wet : float
        water activity of the 'wet' aerosol extinction
    refractive_index_dry : float optional
        refractive index of the dry aerosol
    water_refractive_index : float optional
        refractive index of water
    wavelength : float optional
        wavelength of light in nm
    discretize_Mie : bool optional
        discretize the Mie calculation so it can be cached
    kappa_bounds : tuple optional
        bounds for the kappa value
    kappa_tolerance : float optional
        tolerance for the kappa value
    kappa_maxiter : int optional
        maximum number of iterations for the kappa fit

    Returns
    -------
    kappa : float
        kappa parameter
    """

    def objective_function(kappa_guess):  # negative of the activity
        ratio_guess = extinction_ratio_wet_dry(
            kappa_guess,
            particle_counts=particle_counts,
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
        # sourcery skip
        return np.abs(ratio_guess - Bext_wet / Bext_dry)

    out = fminbound(
        objective_function,
        x1=kappa_bounds[0],
        x2=kappa_bounds[1],
        xtol=kappa_tolerance,
        maxfun=kappa_maxiter,
        full_output=True,
        disp=0
    )
    return out[0]


@lru_cache(maxsize=100000)
def discretize_ScatteringFunction(
        m,
        wavelength,
        diameter,
        minAngle=0,
        maxAngle=180,
        angularResolution=1
):  # sourcery skip
    """
    Discretize the scattering function for a given aerosol size and refractive
    index. This is used to speed up the calculation of the scattering function.
    The function is cached so it can be used multiple times without
    recalculating.

    Args
    ----------
    m : float
        refractive index of the aerosol
    wavlength : float
        wavelength of light in nm
    diameter : float
        diameter of the aerosol in nm
    minAngle : float optional
        minimum angle to calculate the scattering function
    maxAngle : float optional
        maximum angle to calculate the scattering function
    angularResolution : float optional
        angular resolution to calculate the scattering function

    Returns
    -------
    measure : array_like
        scattering function
    SL : array_like
        scattering function
    SR : array_like
        scattering function
    SU : array_like
        scattering function
    """
    measure, SL, SR, SU = ps.ScatteringFunction(
        m,
        wavelength,
        diameter,
        minAngle=minAngle,
        maxAngle=maxAngle,
        angularResolution=angularResolution
    )
    return measure, SL, SR, SU


@lru_cache(maxsize=100000)
def trunc_mono(
        m,
        diameter,
        wavelength=450,
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
    # diameter=list(diameter)

    # constants for the geometry of the CAPS
    diam_sphere = 10.0  # diameter of integrating tube in CAPS in cm
    diam_tube = 1.0  # diameter of tube in CAPS in cm
    extra_length = 0.6  # length outside both sides of integrating sphere
    # calculation is performed (cm)
    z1 = -0.5 * diam_sphere - extra_length  # lower limit of z-axis integral
    z2 = 0.5 * diam_sphere + extra_length  # upper limit of z-axis integral
    npos = 100
    angRes = 0.2

    trunc_calibration = 1.02245612148504  # truncation at calibration diameter
    # of 150nm trunc_calibration value seems to be dependent on angular
    # resolution of Scattering Function and positional resolution of z_axis.

    size_param = (np.pi * diameter) / wavelength  # size parameter

    if discretize:
        m_real = convert.round_arbitrary(np.real(m), base=0.002, mode='round')
        m_imag = convert.round_arbitrary(np.imag(m), base=0.002, mode='round')
        if m_imag == 0:
            m_discretized = m_real
        else:
            m_discretized = m_real + 1j * m_imag
        wavelength_discretized = convert.round_arbitrary(
            wavelength,
            base=1,
            mode='round'
        )
        dp_discretized = float(convert.round_arbitrary(
            diameter,
            base=5,
            mode='round',
            nonzero_edge=True
        ))

        theta, _, _, su = discretize_ScatteringFunction(
            m_discretized,
            wavelength_discretized,
            dp_discretized,
            minAngle=0,
            maxAngle=180,
            angularResolution=angRes
        )

        # theta, _, _, su = ps.ScatteringFunction(
        #     m,
        #     wavelength,
        #     diameter,
        #     minAngle=0,
        #     maxAngle=180,
        #     angularResolution=angRes
        #     )
    else:
        theta, _, _, su = ps.ScatteringFunction(
            m,
            wavelength,
            diameter,
            minAngle=0,
            maxAngle=180,
            angularResolution=angRes
        )
        # Do not set angular resolution higher than 0.1

    # q_mie = ps.MieQ(n, wavelength, diameter)#Mie efficiencies, 0-360 degrees
    # q_mie to high for <100nm particles in the center of integrating sphere.
    # Must integrate su instead
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
    refractive_index,
    diameter_array,
    calibration_diameter,
    wavelength,
    discretize=True
):  # sourcery skip
    """
    Calculate the truncation correction for a given refractive index and
    diameter array.

    Args
    ----------
    refractive_index : float
        Refractive index of the particle.
    diameter_array : array_like
        Array of particle diameters.
    calibration_diameter : float
        Diameter of the calibration sphere.
    wavelength : float
        Wavelength of the light source.

    Returns
    -------
    truncation_array : array_like
        Array of truncation corrections.
    """

    truncation_array = np.zeros(len(diameter_array))

    # self_cal = trunc_mono(
    #     refractive_index,
    #     calibration_diameter,
    #     wavelength=wavelength,
    #     fullOutput=False,
    #     calTrunc=False,
    #     discretize=discretize
    #     )

    for i, diameter in enumerate(diameter_array):
        if discretize:
            m_real = convert.round_arbitrary(
                np.real(refractive_index),
                base=0.002,
                mode='round'
            )
            m_imag = convert.round_arbitrary(
                np.imag(refractive_index),
                base=0.002,
                mode='round'
            )
            if m_imag == 0:
                refractive_index = m_real
            else:
                refractive_index = m_real + 1j * m_imag
            wavelength = convert.round_arbitrary(
                wavelength,
                base=1,
                mode='round'
            )
            diameter = float(convert.round_arbitrary(
                diameter,
                base=20,
                mode='round',
                nonzero_edge=True
            ))
        else:
            pass

        truncation_array[i] = trunc_mono(
            refractive_index,
            diameter,
            wavelength=wavelength,
            fullOutput=False,
            calTrunc=False,
            discretize=True
        )
    return truncation_array  # /self_cal


def bsca_correction_for_distribution_measurements(
    refractive_index,
    diameter_array,
    particle_counts,
    calibration_diameter,
    wavelength,
    discretize=True
):  # sourcery skip
    """
    Calculate the truncation correction for a given refractive index and
    size distribution.

    Args
    ----------
    refractive_index : float
        Refractive index of the particle.
    diameter_array : array_like
        Array of particle diameters.
    particle_counts : array_like
        Array of particle concentrations in #/cm^3.
    calibration_diameter : float
        Diameter of the calibration PSLs. (The SSA normalization point)
    wavelength : float
        Wavelength of the light source.
    discretize : bool, optional
        If True, the calculation will be done with discretized values of
        refractive index, wavelength, and diameter. (Default is True)

    Returns
    -------
    Bsca_correction : float
        Truncation correction. (dimensionless) Multiply the measured Bscat by
        this number to correct for turncation.
        Bsca_corrected = Bsca_measured * Bsca_correction
    """
    trunc_corr = truncation_for_diameters(
        refractive_index,
        diameter_array,
        calibration_diameter,
        wavelength=wavelength,
        discretize=discretize
    )

    # calculate the truncation Bsca
    _, Bsca_trunc, _, _, _, _, _ = Mie_SD(
        refractive_index,
        wavelength,
        dp=diameter_array,
        ndp=particle_counts,
        SMPS=True,
        discretize=discretize,
        truncation_calculation=True,
        truncation_Bsca_multiple=1 / trunc_corr
    )

    # calculate the ideal Bsca
    _, Bsca_ideal, _, _, _, _, _ = Mie_SD(
        refractive_index,
        wavelength,
        dp=diameter_array,
        ndp=particle_counts,
        SMPS=True,
        discretize=discretize,
    )

    return Bsca_ideal / Bsca_trunc


def bsca_correction_for_humidified_measurements(
        kappa,
        particle_counts,
        diameters,
        water_activity_sizer,
        water_activity_sample,
        refractive_index_dry=1.45,
        water_refractive_index=1.33,
        wavelength=450,
        discretize=True,
        calibration_diameter=150,
):  # sourcery skip
    """
    Calculate the truncation correction for a given refractive index and
    size distribution. This function is for humidified measurements.
    Need to provide the kappa values for particles.

    Args
    ----------
    kappa : float
        Kappa value of the particle material.
    particle_counts : array_like
        Array of particle counts in #/cm^3.
    diameters : array_like
        Array of particle diameters.
    water_activity_sizer : float
        Water activity (RH/100) of the sizer air sample.
    water_activity_sample : float
        Water activity (RH/100) of the sample air in the optical measurements.
    refractive_index_dry : float, optional
        Refractive index of the dry particle material. (Default is 1.45)
    water_refractive_index : float, optional
        Refractive index of water. (Default is 1.33)
    wavelength : float, optional
        Wavelength of the light source. (Default is 450)
    discretize : bool, optional
        If True, the calculation will be done with discretized values of
        refractive index, wavelength, and diameter. (Default is True)

    Returns
    -------
    Bsca_correction : float
        Truncation correction. (dimensionless) Multiply the measured Bscat by
        this number to correct for turncation.
        Bsca_corrected = Bsca_measured * Bsca_correction
    """

    # calculate the volume of the dry aerosol
    volume_sizer = convert.length_to_volume(diameters, length_type='diameter')

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
    n_effective = convert.effective_refractive_index(
        refractive_index_dry,
        water_refractive_index,
        volume_water_sample[-1],
        volume_dry[-1]
    )

    # calculate the Bsca correction
    bsca_correction = bsca_correction_for_distribution_measurements(
        n_effective,
        diameter_array=convert.volume_to_length(
            volume_dry + volume_water_sample,
            length_type='diameter'
        ),
        particle_counts=particle_counts,
        calibration_diameter=calibration_diameter,
        wavelength=wavelength,
        discretize=discretize
    )
    return bsca_correction


def kappa_fitting_caps_data(
        datalake,
        truncation_bsca=True,
        refractive_index=1.45,
):  # sourcery skip
    """Fit the extinction ratio with kappa

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
    kappa_fit = np.zeros(
        (len(datalake.datastreams['smps_1D'].return_data(
            keys=['Relative_Humidity_(%)'])[0]), 3),
        dtype=float
    )
    bsca_truncation_dry = np.zeros(len(kappa_fit), dtype=float)
    bsca_truncation_wet = np.zeros(len(kappa_fit), dtype=float)

    for i in tqdm(range(len(kappa_fit))):
        caps_dry = datalake.datastreams['CAPS_data'].return_data(
            keys=[
                'Bext_dry_CAPS_450nm[1/Mm]',
                'dualCAPS_inlet_RH[%]'
            ]
        )[:, i]
        caps_wet = datalake.datastreams['CAPS_data'].return_data(
            keys=[
                'Bext_wet_CAPS_450nm[1/Mm]',
                'Wet_RH_preCAPS[%]',
                'Wet_RH_postCAPS[%]'
            ]
        )[:, i]
        sizer_diameter = np.array(
            datalake.datastreams['smps_2D'].return_header_list()
        ).astype(float)
        sizer_dndlogdp = np.nan_to_num(
            datalake.datastreams['smps_2D'].return_data()[:, i])
        sizer_humidity = datalake.datastreams['smps_1D'].return_data(
            keys=['Relative_Humidity_(%)'])[0, i]
        sizer_total_n = datalake.datastreams['smps_1D'].return_data(
            keys=['Total_Conc_(#/cc)'])[0, i]

        sizer_dn = convert.convert_sizer_dn(sizer_diameter, sizer_dndlogdp)
        sizer_dn = sizer_dn * sizer_total_n / np.sum(sizer_dn)

        if np.isnan(caps_dry).any():
            blank_value = True
        elif np.isnan(caps_wet).any():
            blank_value = True
        elif np.isnan(sizer_diameter).any():
            blank_value = True
        elif np.isnan(sizer_humidity).any():
            blank_value = True
        elif np.sum(np.isnan(sizer_dn)) == len(sizer_dn):
            blank_value = True
        else:
            blank_value = False

        if blank_value:
            kappa_fit[i, :] = np.nan
            if truncation_bsca:
                bsca_truncation_dry[i] = np.nan
                bsca_truncation_wet[i] = np.nan
        else:
            kappa_fit[i, 0] = fit_extinction_ratio_with_kappa(
                Bext_dry=caps_dry[0],
                Bext_wet=caps_wet[0],
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
                Bext_dry=caps_dry[0],
                Bext_wet=caps_wet[0],
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
                Bext_dry=caps_dry[0],
                Bext_wet=caps_wet[0],
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

    # index for Bsca wet and dry
    index_dic = datalake.datastreams['CAPS_data'].return_header_dict()
    index_dic = datalake.datastreams['CAPS_data'].return_header_dict()

    # check if raw in dict
    if 'raw_Bsca_dry_CAPS_450nm[1/Mm]' in index_dic:
        pass
    else:
        # save raw data
        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=datalake.datastreams['CAPS_data'].data_stream[
                index_dic['Bsca_wet_CAPS_450nm[1/Mm]'], :],
            time_new=time,
            header_new=['raw_Bsca_wet_CAPS_450nm[1/Mm]'],
        )
        datalake.datastreams['CAPS_data'].add_processed_data(
            data_new=datalake.datastreams['CAPS_data'].data_stream[
                index_dic['Bsca_dry_CAPS_450nm[1/Mm]'], :],
            time_new=time,
            header_new=['raw_Bsca_dry_CAPS_450nm[1/Mm]'],
        )
        index_dic = datalake.datastreams['CAPS_data'].return_header_dict()

    datalake.datastreams['CAPS_data'].data_stream[
        index_dic['Bsca_wet_CAPS_450nm[1/Mm]'], :] = \
        datalake.datastreams['CAPS_data'].data_stream[
            index_dic['raw_Bsca_wet_CAPS_450nm[1/Mm]'], :] \
        * bsca_truncation_wet.T * calibration_wet

    datalake.datastreams['CAPS_data'].data_stream[
        index_dic['Bsca_dry_CAPS_450nm[1/Mm]'], :] = \
        datalake.datastreams['CAPS_data'].data_stream[
            index_dic['raw_Bsca_dry_CAPS_450nm[1/Mm]'], :] \
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
        keys=['Bsca_wet_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_wet_CAPS_450nm[1/Mm]'])[0]
    ssa_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_dry_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_dry_CAPS_450nm[1/Mm]'])[0]
    ssa_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_wet_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_wet_CAPS_450nm[1/Mm]'])[0]
    ssa_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_dry_CAPS_450nm[1/Mm]'])[0] \
        / datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_dry_CAPS_450nm[1/Mm]'])[0]

    babs_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_wet_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_wet_CAPS_450nm[1/Mm]'])[0]
    babs_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_dry_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_dry_CAPS_450nm[1/Mm]'])[0]
    babs_wet = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_wet_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_wet_CAPS_450nm[1/Mm]'])[0]
    babs_dry = datalake.datastreams['CAPS_data'].return_data(
        keys=['Bext_dry_CAPS_450nm[1/Mm]'])[0] \
        - datalake.datastreams['CAPS_data'].return_data(
        keys=['Bsca_dry_CAPS_450nm[1/Mm]'])[0]

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
        header_new=['Babs_wet_CAPS_450nm[1/Mm]'],
    )
    datalake.datastreams['CAPS_data'].add_processed_data(
        data_new=babs_dry,
        time_new=time,
        header_new=['Babs_dry_CAPS_450nm[1/Mm]'],
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
    # index for Bsca wet and dry
    index_dic = datalake.datastreams['pass3'].return_header_dict()
    time = datalake.datastreams['pass3'].return_time(
        datetime64=False,
        raw=True
    )
    babs_list = ['Babs405nm[1/Mm]', 'Babs532nm[1/Mm]', 'Babs781nm[1/Mm]']
    bsca_list = ['Bsca405nm[1/Mm]', 'Bsca532nm[1/Mm]', 'Bsca781nm[1/Mm]']

    if 'raw_Babs405nm[1/Mm]' not in index_dic:
        print('Copy raw babs Pass-3')
        for babs in babs_list:
            raw_name = f'raw_{babs}'

            datalake.datastreams['pass3'].add_processed_data(
                data_new=datalake.datastreams['pass3'].data_stream[
                    index_dic[babs], :],
                time_new=time,
                header_new=[raw_name],
            )
    if 'raw_Bsca405nm[1/Mm]' not in index_dic:
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
