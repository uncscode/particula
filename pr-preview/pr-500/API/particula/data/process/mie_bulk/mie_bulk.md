# Mie Bulk

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Mie Bulk

> Auto-generated documentation for [particula.data.process.mie_bulk](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_bulk.py) module.

## compute_bulk_optics

[Show source in mie_bulk.py:129](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_bulk.py#L129)

Compute bulk optical properties from size-dependent efficiency factors for
a size distribution.

This function calculates various bulk optical properties such as
extinction, scattering, and backscattering coefficients based on the size
distribution and corresponding efficiency factors.

#### Arguments

- `q_ext` - Array of extinction efficiency factors.
- `q_sca` - Array of scattering efficiency factors.
- `q_back` - Array of backscatter efficiency factors.
- `q_ratio` - Array of backscatter-to-extinction ratio efficiency factors.
- `g` - Array of asymmetry factors.
- `area_dist` - Area-scaled size distribution array.
- `extinction_only` - Flag indicating whether to compute only the
    extinction coefficient.
- `pms` - Flag indicating whether a probability mass distribution is used,
    where the sum of all bins
    represents the total number of particles.
- `dp` - Array of particle diameters in nanometers.

#### Returns

If `extinction_only` is True, returns an array with the bulk
extinction coefficient. Otherwise, returns a tuple containing the
bulk optical properties, including extinction, scattering, and
backscattering coefficients, and possibly others depending on
the input flags.

#### Signature

```python
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
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], ...]]: ...
```



## discretize_auto_mieq

[Show source in mie_bulk.py:19](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_bulk.py#L19)

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

#### Arguments

- `m_sphere` - The complex refractive index of the sphere. A real
    number can be provided for non-absorbing materials.
- `wavelength` - The wavelength of the incident light in nanometers (nm).
- `diameter` - The diameter of the sphere in nanometers (nm).
- `m_medium` - The refractive index of the surrounding medium.
    Default is 1.0, corresponding to vacuum.

#### Returns

Tuple:
    - q_ext, Extinction efficiency.
    - q_sca, Scattering efficiency.
    - q_abs, Absorption efficiency.
    - g, Asymmetry factor.
    - q_pr, Radiation pressure efficiency.
    - q_back, Backscatter efficiency.
    - q_ratio, Ratio of backscatter to extinction efficiency.

#### Signature

```python
@lru_cache(maxsize=100000)
def discretize_auto_mieq(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: float,
    m_medium: float = 1.0,
) -> Tuple[float, ...]: ...
```



## discretize_mie_parameters

[Show source in mie_bulk.py:64](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_bulk.py#L64)

Discretize the refractive index, wavelength, and diameters for Mie
scattering calculations.

This function improves numerical stability and performance by discretizing
the refractive index of the material, the wavelength of incident light,
and the diameters of particles. Discretization reduces the variability
in input parameters, making Mie scattering computations more efficient
by creating a more manageable set of unique calculations.

#### Arguments

- `m_sphere` - The complex or real refractive index of the particles.
    This value is discretized to a specified base to reduce
    input variability.
- `wavelength` - The wavelength of incident light in nanometers (nm),
    discretized to minimize variations in related computations.
- `diameter` - The particle diameter or array of diameters in nanometers
    (nm), discretized to a specified base to standardize input sizes
    for calculations.
- `base_m_sphere` - Optional; the base value to which the real and
    imaginary parts of the refractive index are rounded.
    Default is 0.001.
- `base_wavelength` - Optional; the base value to which the wavelength is
    rounded. Default is 1 nm.
- `base_diameter` - Optional; the base value to which particle diameters
    are rounded. Default is 5 nm.

#### Returns

Tuple:
    - The discretized refractive index (m_sphere).
    - The discretized wavelength.
    - The discretized diameter or array of diameters, suitable for use
        in Mie scattering calculations with potentially improved
        performance and reduced computational overhead.

#### Signature

```python
def discretize_mie_parameters(
    m_sphere: Union[complex, float],
    wavelength: float,
    diameter: Union[float, NDArray[np.float64]],
    base_m_sphere: float = 0.001,
    base_wavelength: float = 1,
    base_diameter: float = 5,
) -> Tuple[Union[complex, float], float, Union[float, list[float]]]: ...
```



## format_mie_results

[Show source in mie_bulk.py:194](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_bulk.py#L194)

Format the output results of the Mie scattering calculations.

#### Arguments

- `b_ext` - Array of bulk extinction coefficients.
- `b_sca` - Array of bulk scattering coefficients.
- `b_abs` - Array of bulk absorption coefficients.
- `big_g` - Array of asymmetry factors (g).
- `b_pr` - Array of bulk radiation pressure efficiencies.
- `b_back` - Array of bulk backscattering coefficients.
- `b_ratio` - Array of backscatter-to-extinction ratios.
- `as_dict` - Flag to determine if the results should be returned as a
    dictionary.

#### Returns

(dict, Tuple):
    - If `as_dict` is True, returns a dictionary with the bulk optical
        properties.
    - If `as_dict` is False, returns a tuple of the bulk
        optical properties in the following order,
        (b_ext, b_sca, b_abs, big_g, b_pr, b_back, b_ratio).

#### Signature

```python
def format_mie_results(
    b_ext: NDArray[np.float64],
    b_sca: NDArray[np.float64],
    b_abs: NDArray[np.float64],
    big_g: NDArray[np.float64],
    b_pr: NDArray[np.float64],
    b_back: NDArray[np.float64],
    b_ratio: NDArray[np.float64],
    as_dict: bool,
) -> Union[dict[str, NDArray[np.float64]], tuple[NDArray[np.float64], ...]]: ...
```



## mie_size_distribution

[Show source in mie_bulk.py:239](https://github.com/uncscode/particula/blob/main/particula/data/process/mie_bulk.py#L239)

Calculate Mie scattering parameters for a size distribution of spherical
particles.

This function computes optical properties such as extinction, scattering,
absorption coefficients, asymmetry factor, backscatter efficiency, and
their ratios for a given size distribution of spherical particles. It
supports various modes of calculation, including discretization of input
parameters and optional truncation of the scattering efficiency.

#### Arguments

- `m_sphere` - The complex refractive index of the particles. Real values
    can be used for non-absorbing materials.
- `wavelength` - The wavelength of the incident light in nanometers (nm).
- `diameter` - An array of particle diameters in nanometers (nm).
- `number_per_cm3` - The number distribution of particles per cubic
    centimeter (#/cm^3).
- `n_medium` - The refractive index of the medium. Defaults to 1.0
    (air or vacuum).
- `pms` - Specifies if the size distribution is in probability mass form.
    Default is True.
- `as_dict` - If True, results are returned as a dictionary. Otherwise,
    as a tuple. Default is False.
- `extinction_only` - If True, only the extinction coefficient is
    calculated and returned. Default is False.
- `discretize` - If True, input parameters (m_sphere, wavelength, diameter)
    are discretized for computation. Default is False.
- `truncation_calculation` - Enables truncation of the scattering
    efficiency based on a multiple of the backscattering coefficient.
    Default is False.
- `truncation_b_sca_multiple` - The multiple of the backscattering
    coefficient used for truncating the scattering efficiency.
    Required if `truncation_calculation` is True.

#### Returns

(NDArray, dict, Tuple):
    - An array of extinction coefficients if `extinction_only` is True.
    - A dictionary of computed optical properties if `as_dict` is True.
    - A tuple of computed optical properties otherwise.

#### Raises

- `ValueError` - If `truncation_calculation` is True but
    `truncation_b_sca_multiple` is not specified.

#### Signature

```python
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
    NDArray[np.float64], dict[str, NDArray[np.float64]], Tuple[NDArray[np.float64], ...]
]: ...
```
