"""Calculate Kappa hygroscopic parameters from extinction data."""
# pyright: reportReturnType=false, reportAssignmentType=false
# pyright: reportIndexIssue=false
# pyright: reportArgumentType=false, reportOperatorIssue=false
# pylint: disable=too-many-arguments, too-many-locals


from typing import Union, Tuple
import numpy as np
from scipy.optimize import fminbound
from numpy.typing import NDArray
from particula.util import convert
from particula.data.process import mie_bulk


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
    discretize: bool = True,
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
    discretize : bool, optional
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

    optics_dry = mie_bulk.mie_size_distribution(
        m_sphere=n_effective_dry,
        wavelength=wavelength,
        diameter=diameters_dry,  # pyright: ignore[reportArgumentType]
        number_per_cm3=number_per_cm3,
        pms=True,
        extinction_only=not return_all_optics,
        discretize=discretize)

    optics_wet = mie_bulk.mie_size_distribution(
        m_sphere=n_effective_wet,
        wavelength=wavelength,
        diameter=diameters_wet,  # pyright: ignore[reportArgumentType]
        number_per_cm3=number_per_cm3,
        pms=True,
        extinction_only=not return_all_optics,
        discretize=discretize)

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
    discretize: bool = True,
    kappa_bounds: Tuple[float, float] = (0, 1),
    kappa_tolerance: float = 1e-6,
    kappa_maxiter: int = 200,
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
    discretize : bool, optional
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
            discretize=discretize,
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


def kappa_from_extinction_looped(
    extinction_dry: NDArray[np.float64],
    extinction_wet: NDArray[np.float64],
    number_per_cm3: NDArray[np.float64],
    diameter: NDArray[np.float64],
    water_activity_sizer: NDArray[np.float64],
    water_activity_sample_dry: NDArray[np.float64],
    water_activity_sample_wet: NDArray[np.float64],
    refractive_index_dry: Union[complex, float] = 1.45,
    water_refractive_index: Union[complex, float] = 1.33,
    wavelength: float = 450,
    discretize: bool = True
) -> NDArray[np.float64]:
    """
    Fits the extinction ratio to the kappa value for a given set of
    measurements, looping over time indexes in number_per_cm3. This function
    is tailored for analyzing data from a CAPS (Cavity Attenuated Phase Shift)
    instrument under varying humidities.

    Parameters
    ----------
    extinction_dry : NDArray[np.float64]
        Array of dry aerosol extinction measurements.
    extinction_wet : NDArray[np.float64]
        Array of wet aerosol extinction measurements.
    number_per_cm3 : NDArray[np.float64]
        Array of particle number concentrations in #/cm³.
    diameter : NDArray[np.float64]
        Array of particle diameters.
    water_activity_sizer : NDArray[np.float64]
        Water activity (relative humidity/100) of the sizing instrument's air.
    water_activity_sample_dry : NDArray[np.float64]
        Water activity (relative humidity/100) of the air for dry measurements.
    water_activity_sample_wet : NDArray[np.float64]
        Water activity (relative humidity/100) of the air for wet measurements.
    refractive_index_dry : Union[complex, float], optional
        Refractive index of dry particles. Default is 1.45.
    water_refractive_index : Union[complex, float], optional
        Refractive index of water. Default is 1.33.
    wavelength : float, optional
        Wavelength of the light source in nanometers. Default is 450.
    discretize : bool, optional
        If True, calculations are performed with discretized parameter values
        to potentially improve performance. Default is True.

    Returns
    -------
    NDArray[np.float64]
        A 2D array where each row corresponds to the time-indexed kappa value,
        lower and upper bounds of the kappa estimation, structured as
        [kappa, lower, upper].
    """
    # Configuration for kappa fitting
    kappa_bounds = (0, 1.5)
    kappa_tolerance = 1e-6
    kappa_maxiter = 250
    upper_scale = 1.05
    lower_scale = 0.95

    # Initialize the array for fitting results
    kappa_fit = np.zeros((len(extinction_wet), 3), dtype=np.float64)

    # Combine NaN checks for input arrays
    skip_data = np.isnan(extinction_dry) | np.isnan(extinction_wet) | \
        np.isnan(water_activity_sample_wet) | \
        np.isnan(water_activity_sizer) | \
        np.isnan(water_activity_sample_dry) | \
        np.isnan(number_per_cm3).any(axis=1)

    # Loop over each time index
    for index, row_number_per_cm3 in enumerate(number_per_cm3):
        if skip_data[index]:
            kappa_fit[index, :] = np.nan  # Assign NaN if any data is missing
            continue
        # Perform kappa fitting for the current index
        kappa_fit[index, 0] = fit_extinction_ratio_with_kappa(
            b_ext_dry=extinction_dry[index],
            b_ext_wet=extinction_wet[index],
            number_per_cm3=row_number_per_cm3,
            diameters=diameter,
            water_activity_sizer=water_activity_sizer[index],
            water_activity_dry=water_activity_sample_dry[index],
            water_activity_wet=water_activity_sample_wet[index],
            refractive_index_dry=refractive_index_dry,
            water_refractive_index=water_refractive_index,
            wavelength=wavelength,
            discretize=discretize,
            kappa_bounds=kappa_bounds,
            kappa_tolerance=kappa_tolerance,
            kappa_maxiter=kappa_maxiter,
        )
        # lower kappa value, using a higher dry refractive index
        kappa_fit[index, 1] = fit_extinction_ratio_with_kappa(
            b_ext_dry=extinction_dry[index],
            b_ext_wet=extinction_wet[index],
            number_per_cm3=row_number_per_cm3,
            diameters=diameter,
            water_activity_sizer=water_activity_sizer[index],
            water_activity_dry=water_activity_sample_dry[index],
            water_activity_wet=water_activity_sample_wet[index],
            refractive_index_dry=refractive_index_dry * upper_scale,
            water_refractive_index=water_refractive_index,
            wavelength=wavelength,
            discretize=discretize,
            kappa_bounds=kappa_bounds,
            kappa_tolerance=kappa_tolerance,
            kappa_maxiter=kappa_maxiter,
        )
        # upper kappa value, using a lower dry refractive index
        kappa_fit[index, 2] = fit_extinction_ratio_with_kappa(
            b_ext_dry=extinction_dry[index],
            b_ext_wet=extinction_wet[index],
            number_per_cm3=row_number_per_cm3,
            diameters=diameter,
            water_activity_sizer=water_activity_sizer[index],
            water_activity_dry=water_activity_sample_dry[index],
            water_activity_wet=water_activity_sample_wet[index],
            refractive_index_dry=refractive_index_dry * lower_scale,
            water_refractive_index=water_refractive_index,
            wavelength=wavelength,
            discretize=discretize,
            kappa_bounds=kappa_bounds,
            kappa_tolerance=kappa_tolerance,
            kappa_maxiter=kappa_maxiter,
        )
    return kappa_fit
