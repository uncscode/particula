# Kappa Via Extinction

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Kappa Via Extinction

> Auto-generated documentation for [particula.data.process.kappa_via_extinction](https://github.com/uncscode/particula/blob/main/particula/data/process/kappa_via_extinction.py) module.

## extinction_ratio_wet_dry

[Show source in kappa_via_extinction.py:18](https://github.com/uncscode/particula/blob/main/particula/data/process/kappa_via_extinction.py#L18)

Calculate the extinction ratio between wet and dry aerosols, considering
water uptake through the kappa parameter.

This function uses Mie theory to determine the optical properties of
aerosols with varying water content, allowing for analysis of hygroscopic
growth and its impact on aerosol optical characteristics.

#### Arguments

- `kappa` - Hygroscopicity parameter, defining the water uptake ability
    of particles.
- `number_per_cm3` - Number concentration of particles per cubic
    centimeter for each size bin.
- `diameters` - Diameters of particles in nanometers for each size bin.
- `water_activity_sizer` - Water activity of the aerosol size distribution.
- `water_activity_dry` - Water activity for the calculation of "dry"
    aerosol properties.
- `water_activity_wet` - Water activity for the calculation of "wet"
    aerosol properties.
- `refractive_index_dry` - Refractive index of the dry aerosol particles.
    Default is 1.45.
- `water_refractive_index` - Refractive index of water. Default is 1.33.
- `wavelength` - Wavelength of the incident light in nanometers.
    Default is 450 nm.
- `discretize` - If True, discretizes input arguments for Mie calculations
    to enable caching. Default is True.
- `return_coefficients` - If True, returns the individual extinction
    coefficients for wet and dry aerosols instead of their ratio.
    Default is False.
- `return_all_optics` - If True, returns all optical properties calculated
    by Mie theory, not just extinction. Default is False.

#### Returns

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

[Show source in kappa_via_extinction.py:134](https://github.com/uncscode/particula/blob/main/particula/data/process/kappa_via_extinction.py#L134)

Fit the kappa parameter based on the measured extinction ratios of dry
and wet aerosols, considering water uptake effects.

This method uses Mie theory to optimize kappa by minimizing the difference
between the calculated and observed extinction ratios of wet to dry
aerosols.

#### Arguments

- `b_ext_dry` - The measured extinction of the dry aerosol.
- `b_ext_wet` - The measured extinction of the wet aerosol.
- `number_per_cm3` - Number concentration of particles per cubic centimeter
    for each size bin.
- `diameters` - Diameters of particles in nanometers for each size bin.
- `water_activity_sizer` - Water activity corresponding to the aerosol
    size distribution.
- `water_activity_dry` - Water activity for the "dry" aerosol condition.
- `water_activity_wet` - Water activity for the "wet" aerosol condition.
- `refractive_index_dry` - Refractive index of the dry aerosol particles.
    Default is 1.45.
- `water_refractive_index` - Refractive index of water. Default is 1.33.
- `wavelength` - Wavelength of incident light in nanometers. Default is
    450 nm.
- `discretize` - If True, discretizes input arguments for Mie calculations
    to enable caching. Default is True.
- `kappa_bounds` - Bounds within which to fit the kappa parameter.
    Default is (0, 1).
- `kappa_tolerance` - Tolerance level for the optimization of kappa.
    Default is 1e-6.
- `kappa_maxiter` - Maximum number of iterations allowed in the optimization
    process. Default is 200.

#### Returns

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

[Show source in kappa_via_extinction.py:228](https://github.com/uncscode/particula/blob/main/particula/data/process/kappa_via_extinction.py#L228)

Fit the extinction ratio to the kappa value for a set of measurements,
looping over time indexes in `number_per_cm3`.

This function is designed for analyzing data from a CAPS (Cavity Attenuated
Phase Shift) instrument under varying humidities.

#### Arguments

- `extinction_dry` - Array of dry aerosol extinction measurements.
- `extinction_wet` - Array of wet aerosol extinction measurements.
- `number_per_cm3` - Array of particle number concentrations in #/cm³.
- `diameter` - Array of particle diameters in nanometers.
- `water_activity_sizer` - Water activity (relative humidity/100) of the
    sizing instrument's air.
- `water_activity_sample_dry` - Water activity (relative humidity/100) of
    the air for dry measurements.
- `water_activity_sample_wet` - Water activity (relative humidity/100) of
    the air for wet measurements.
- `refractive_index_dry` - Refractive index of dry particles.
    Default is 1.45.
- `water_refractive_index` - Refractive index of water. Default is 1.33.
- `wavelength` - Wavelength of the light source in nanometers.
    Default is 450 nm.
- `discretize` - If True, calculations are performed with discretized
    parameter values to potentially improve performance.
    Default is True.

#### Returns

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
