# Scattering Truncation

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Scattering Truncation

> Auto-generated documentation for [particula.data.process.scattering_truncation](https://github.com/Gorkowski/particula/blob/main/particula/data/process/scattering_truncation.py) module.

## correction_for_distribution

[Show source in scattering_truncation.py:246](https://github.com/Gorkowski/particula/blob/main/particula/data/process/scattering_truncation.py#L246)

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

#### Signature

```python
def correction_for_distribution(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter_sizes: NDArray[np.float64],
    number_per_cm3: NDArray[np.float64],
    discretize: bool = True,
) -> Union[float, np.float64]: ...
```



## correction_for_humidified

[Show source in scattering_truncation.py:321](https://github.com/Gorkowski/particula/blob/main/particula/data/process/scattering_truncation.py#L321)

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

#### Signature

```python
def correction_for_humidified(
    kappa: Union[float, np.float64],
    number_per_cm3: NDArray[np.float64],
    diameter: NDArray[np.float64],
    water_activity_sizer: np.float64,
    water_activity_sample: np.float64,
    refractive_index_dry: Union[complex, float] = 1.45,
    water_refractive_index: Union[complex, float] = 1.33,
    wavelength: float = 450,
    discretize: bool = True,
) -> np.float64: ...
```



## correction_for_humidified_looped

[Show source in scattering_truncation.py:407](https://github.com/Gorkowski/particula/blob/main/particula/data/process/scattering_truncation.py#L407)

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

#### Signature

```python
def correction_for_humidified_looped(
    kappa: NDArray[np.float64],
    number_per_cm3: NDArray[np.float64],
    diameter: NDArray[np.float64],
    water_activity_sizer: NDArray[np.float64],
    water_activity_sample: NDArray[np.float64],
    refractive_index_dry: Union[complex, float] = 1.45,
    water_refractive_index: Union[complex, float] = 1.33,
    wavelength: float = 450,
    discretize: bool = True,
) -> NDArray[np.float64]: ...
```



## get_truncated_scattering

[Show source in scattering_truncation.py:16](https://github.com/Gorkowski/particula/blob/main/particula/data/process/scattering_truncation.py#L16)

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

#### Signature

```python
def get_truncated_scattering(
    scattering_unpolarized: np.ndarray, theta: np.ndarray, theta1: float, theta2: float
) -> Tuple[np.ndarray, np.ndarray]: ...
```



## trunc_mono

[Show source in scattering_truncation.py:51](https://github.com/Gorkowski/particula/blob/main/particula/data/process/scattering_truncation.py#L51)

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

#### Signature

```python
@lru_cache(maxsize=100000)
def trunc_mono(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: float,
    full_output: bool = False,
    calibrated_trunc: bool = True,
    discretize: bool = True,
) -> Union[
    float, Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]: ...
```



## truncation_for_diameters

[Show source in scattering_truncation.py:189](https://github.com/Gorkowski/particula/blob/main/particula/data/process/scattering_truncation.py#L189)

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

#### Signature

```python
def truncation_for_diameters(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter_sizes: NDArray[np.float64],
    discretize: bool = True,
    calibrated_trunc: bool = True,
) -> NDArray[np.float64]: ...
```
