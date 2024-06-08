# Kappa Via Extinction

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Kappa Via Extinction

> Auto-generated documentation for [particula.data.process.kappa_via_extinction](https://github.com/Gorkowski/particula/blob/main/particula/data/process/kappa_via_extinction.py) module.

## extinction_ratio_wet_dry

[Show source in kappa_via_extinction.py:16](https://github.com/Gorkowski/particula/blob/main/particula/data/process/kappa_via_extinction.py#L16)

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

#### Signature

```python
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
) -> Union[float, Tuple[NDArray, NDArray]]: ...
```



## fit_extinction_ratio_with_kappa

[Show source in kappa_via_extinction.py:131](https://github.com/Gorkowski/particula/blob/main/particula/data/process/kappa_via_extinction.py#L131)

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

#### Signature

```python
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
    kappa_tolerance: float = 1e-06,
    kappa_maxiter: int = 200,
) -> Union[float, np.float64]: ...
```



## kappa_from_extinction_looped

[Show source in kappa_via_extinction.py:232](https://github.com/Gorkowski/particula/blob/main/particula/data/process/kappa_via_extinction.py#L232)

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

#### Signature

```python
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
    discretize: bool = True,
) -> NDArray[np.float64]: ...
```
