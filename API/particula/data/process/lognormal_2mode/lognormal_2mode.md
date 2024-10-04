# Lognormal 2mode

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Lognormal 2mode

> Auto-generated documentation for [particula.data.process.lognormal_2mode](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py) module.

#### Attributes

- `logger` - Set up logging: logging.getLogger('particula')


## cost_function

[Show source in lognormal_2mode.py:26](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L26)

Cost function for the lognormal distribution with 2 modes.

#### Arguments

- `params` - Combined array of mode_values, geometric_standard_deviation,
    and number_of_particles.
- `x_values` - The x-values (particle sizes).
- `concentration_pdf` - The actual concentration PDF to fit.

#### Returns

The mean squared error between the actual and guessed concentration
    PDF.

#### Signature

```python
def cost_function(
    params: NDArray[np.float64],
    x_values: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> float: ...
```



## create_lognormal_2mode_from_fit

[Show source in lognormal_2mode.py:383](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L383)

Create a fitted PMF stream and concentration matrix based on
optimized parameters.

#### Arguments

- `parameters_stream` - Stream object containing the optimized parameters.
- `radius_min` - Log10 of the minimum radius value in meters (default: -9).
- `radius_max` - Log10 of the maximum radius value in meters (default: -6).
- `num_radius_bins` - Number of radius bins to create between radius_min
    and radius_max.

#### Returns

- `fitted_pmf_stream` - A Stream object containing the time and fitted
    concentration PMF data.
- `fitted_concentration_pmf` - A numpy array with the fitted
    concentration PMF values.

#### Signature

```python
def create_lognormal_2mode_from_fit(
    parameters_stream: Stream,
    radius_min: float = 1e-09,
    radius_max: float = 1e-06,
    num_radius_bins: int = 250,
) -> Tuple[Stream, NDArray[np.float64]]: ...
```

#### See also

- [Stream](../stream.md#stream)



## evaluate_fit

[Show source in lognormal_2mode.py:131](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L131)

Evaluate the best fit and calculate R² score.

#### Signature

```python
def evaluate_fit(
    best_result: dict[str, Any],
    logspace_x: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]: ...
```



## get_bounds

[Show source in lognormal_2mode.py:73](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L73)

Provide default bounds for optimization parameters.

#### Signature

```python
def get_bounds() -> List[Tuple[float, Any]]: ...
```



## get_initial_guesses

[Show source in lognormal_2mode.py:85](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L85)

Combine initial guesses into a single array.

#### Signature

```python
def get_initial_guesses(
    mode_guess: NDArray[np.float64],
    geometric_standard_deviation_guess: NDArray[np.float64],
    number_of_particles_in_mode_guess: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## guess_and_optimize_looped

[Show source in lognormal_2mode.py:301](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L301)

Generate initial guesses using a machine learning model, optimize them,
and return a Stream object with the results.

#### Arguments

- `experiment_time` - Array of experiment time points.
- `radius_m` - Array of particle radii in meters.
- `concentration_m3_pdf` - 2D array of concentration PDFs for each
    time point.

#### Returns

- `fitted_stream` - A Stream object containing the initial guesses,
    optimized values, and R² scores.

#### Signature

```python
def guess_and_optimize_looped(
    experiment_time: NDArray[np.float64],
    radius_m: NDArray[np.float64],
    concentration_m3_pdf: NDArray[np.float64],
) -> Stream: ...
```

#### See also

- [Stream](../stream.md#stream)



## optimize_fit

[Show source in lognormal_2mode.py:165](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L165)

Optimize the lognormal 2-mode distribution parameters using multiple
optimization methods.

#### Signature

```python
def optimize_fit(
    mode_guess: NDArray[np.float64],
    geometric_standard_deviation_guess: NDArray[np.float64],
    number_of_particles_in_mode_guess: NDArray[np.float64],
    logspace_x: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
    bounds: Optional[List[Tuple[float, Any]]] = None,
    list_of_methods: Optional[List[str]] = None,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, dict[str, Any]
]: ...
```



## optimize_fit_looped

[Show source in lognormal_2mode.py:232](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L232)

Loop through the concentration PDFs to get the best optimization.

#### Arguments

- `mode_guess` - Array of mode values.
- `geometric_standard_deviation_guess` - Array of geometric standard
    deviations.
- `number_of_particles_in_mode_guess` - Array of number of particles.
- `x_values` - Array of x-values (particle sizes).
- `concentration_pdf` - Matrix of concentration PDF values.
- `bounds` - List of bounds for optimization.
- `list_of_methods` - List of optimization methods.

#### Returns

- `optimized_mode_values` - Optimized mode values.
- `optimized_gsd` - Optimized geometric standard deviations.
- `optimized_number_of_particles` - Optimized number of particles.
- `r2` - R² score.
- `optimization_results` - Dictionary of optimization results.

#### Signature

```python
def optimize_fit_looped(
    mode_guess: NDArray[np.float64],
    geometric_standard_deviation_guess: NDArray[np.float64],
    number_of_particles_in_mode_guess: NDArray[np.float64],
    logspace_x: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
    bounds: Optional[List[Tuple[float, Any]]] = None,
    list_of_methods: Optional[List[str]] = None,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]: ...
```



## run_optimization

[Show source in lognormal_2mode.py:100](https://github.com/uncscode/particula/blob/main/particula/data/process/lognormal_2mode.py#L100)

Perform the optimization using the specified method.

#### Signature

```python
def run_optimization(
    method: str,
    initial_guess: NDArray[np.float64],
    bounds: List[Tuple[float, Any]],
    x_values: NDArray[np.float64],
    concentration_pdf: NDArray[np.float64],
) -> Optional[dict[str, Any]]: ...
```
