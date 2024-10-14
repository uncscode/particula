# Chamber Rate Fitting

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Data](../index.md#data) / [Process](./index.md#process) / Chamber Rate Fitting

> Auto-generated documentation for [particula.data.process.chamber_rate_fitting](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py) module.

## ChamberParameters

[Show source in chamber_rate_fitting.py:174](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L174)

Data class for the chamber parameters.

#### Signature

```python
class ChamberParameters: ...
```



## calculate_optimized_rates

[Show source in chamber_rate_fitting.py:294](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L294)

Calculate the coagulation rates using the optimized parameters and return
the rates and R2 score.

#### Arguments

- `radius_bins` - Array of particle radii in meters.
- `concentration_pmf` - 2D array of concentration PMF values.
- `wall_eddy_diffusivity` - Optimized wall eddy diffusivity.
- `alpha_collision_efficiency` - Optimized alpha collision efficiency.
- `chamber_params` - ChamberParameters object containing chamber-related
    parameters.
- `time_derivative_concentration_pmf` - Array of observed rate of change
    of the concentration PMF (optional).

#### Returns

- `coagulation_loss` - Loss rate due to coagulation.
- `coagulation_gain` - Gain rate due to coagulation.
- `dilution_loss` - Loss rate due to dilution.
- `wall_loss_rate` - Loss rate due to wall deposition.
- `net_rate` - Net rate considering all effects.
- `r2_value` - R2 score between the net rate and the observed rate.

#### Signature

```python
def calculate_optimized_rates(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    wall_eddy_diffusivity: float,
    alpha_collision_efficiency: float,
    chamber_parameters: ChamberParameters,
    time_derivative_concentration_pmf: Optional[NDArray[np.float64]] = None,
) -> Tuple[float, float, float, float, float, float]: ...
```

#### See also

- [ChamberParameters](#chamberparameters)



## calculate_pmf_rates

[Show source in chamber_rate_fitting.py:22](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L22)

Calculate the coagulation, dilution, and wall loss rates,
and return the net rate.

#### Arguments

- `radius_bins` - Array of particle radii.
- `concentration_pmf` - Array of particle concentration
    probability mass function.
- `temperature` - Temperature in Kelvin.
- `pressure` - Pressure in Pascals.
- `particle_density` - Density of the particles in kg/m^3.
- `alpha_collision_efficiency` - Collision efficiency factor.
- `volume` - Volume of the chamber in m^3.
- `input_flow_rate` - Input flow rate in m^3/s.
- `wall_eddy_diffusivity` - Eddy diffusivity for wall loss in m^2/s.
- `chamber_dimensions` - Dimensions of the chamber
    (length, width, height) in meters.

#### Returns

- `coagulation_loss` - Loss rate due to coagulation.
- `coagulation_gain` - Gain rate due to coagulation.
- `dilution_loss` - Loss rate due to dilution.
- `wall_loss_rate` - Loss rate due to wall deposition.
- `net_rate` - Net rate considering all effects.

#### Signature

```python
def calculate_pmf_rates(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    alpha_collision_efficiency: float = 1,
    volume: float = 1,
    input_flow_rate: float = 1.6e-07,
    wall_eddy_diffusivity: float = 0.1,
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]: ...
```



## coagulation_rates_cost_function

[Show source in chamber_rate_fitting.py:122](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L122)

Cost function for the optimization of the eddy diffusivity
and alpha collision efficiency.

#### Signature

```python
def coagulation_rates_cost_function(
    parameters: NDArray[np.float64],
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    time_derivative_concentration_pmf: NDArray[np.float64],
    temperature: float = 293.15,
    pressure: float = 101325,
    particle_density: float = 1000,
    volume: float = 1,
    input_flow_rate: float = 1.6e-07,
    chamber_dimensions: Tuple[float, float, float] = (1, 1, 1),
) -> float: ...
```



## create_guess_and_bounds

[Show source in chamber_rate_fitting.py:185](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L185)

Create the initial guess array and bounds list for the optimization.

#### Arguments

- `guess_eddy_diffusivity` - Initial guess for eddy diffusivity.
- `guess_alpha_collision_efficiency` - Initial guess for alpha collision
    efficiency.
- `bounds_eddy_diffusivity` - Bounds for eddy diffusivity.
- `bounds_alpha_collision_efficiency` - Bounds for alpha collision
    efficiency.

#### Returns

- `initial_guess` - Numpy array of the initial guess values.
- `bounds` - List of tuples representing the bounds for each parameter.

#### Signature

```python
def create_guess_and_bounds(
    guess_eddy_diffusivity: float,
    guess_alpha_collision_efficiency: float,
    bounds_eddy_diffusivity: Tuple[float, float],
    bounds_alpha_collision_efficiency: Tuple[float, float],
) -> Tuple[NDArray[np.float64], List[Tuple[float, float]]]: ...
```



## optimize_and_calculate_rates_looped

[Show source in chamber_rate_fitting.py:363](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L363)

Perform optimization and calculate rates for each time point in the stream.

#### Arguments

- `pmf_stream` - Stream object containing the fitted PMF data.
- `pmf_derivative_stream` - Stream object containing the derivative of the
    PMF data.
- `chamber_parameters` - ChamberParameters object containing
    chamber-related parameters.
- `fit_guess` - Initial guess for the optimization.
- `fit_bounds` - Bounds for the optimization parameters.

#### Returns

- `result_stream` - Stream containing the optimization results for
    each time point.
- `coagulation_loss_stream` - Stream containing coagulation loss rates.
- `coagulation_gain_stream` - Stream containing coagulation gain rates.
- `coagulation_net_stream` - Stream containing net coagulation rates.
- `dilution_loss_stream` - Stream containing dilution loss rates.
- `wall_loss_rate_stream` - Stream containing wall loss rates.
- `total_rate_stream` - Stream containing total rates.

#### Signature

```python
def optimize_and_calculate_rates_looped(
    pmf_stream: Stream,
    pmf_derivative_stream: Stream,
    chamber_parameters: ChamberParameters,
    fit_guess: NDArray[np.float64],
    fit_bounds: List[Tuple[float, float]],
) -> Tuple[Stream, Stream, Stream, Stream, Stream, Stream, Stream]: ...
```

#### See also

- [ChamberParameters](#chamberparameters)
- [Stream](../stream.md#stream)



## optimize_chamber_parameters

[Show source in chamber_rate_fitting.py:232](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L232)

Optimize the eddy diffusivity and alpha collision efficiency parameters
for a given particle size distribution and its time derivative.

#### Arguments

- `radius_bins` - Array of particle size bins in meters.
- `concentration_pmf` - Array of particle mass fractions (PMF)
    concentrations at each radius bin.
- `time_derivative_concentration_pmf` - Array of time derivatives of
    the PMF concentrations, representing the rate of change
    in concentration over time.
- `chamber_params` - ChamberParameters object containing the physical
    properties of the chamber, including temperature, pressure,
    particle density, volume, input flow rate, and chamber dimensions.
- `fit_guess` - Initial guess for the optimization parameters
    (eddy diffusivity and alpha collision efficiency).
- `fit_bounds` - List of tuples specifying the bounds for the
    optimization parameters (lower and upper bounds
    for each parameter).
- `minimize_method` - Optimization method to be used. Default is "L-BFGS-B".
    The following methods from `scipy.optimize.minimize` accept bounds,
    "L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr".

#### Returns

- `wall_eddy_diffusivity_optimized` - Optimized value of the wall eddy
    diffusivity (in 1/s).
- `alpha_collision_efficiency_optimized` - Optimized value of the alpha
    collision efficiency (dimensionless).

#### Signature

```python
def optimize_chamber_parameters(
    radius_bins: NDArray[np.float64],
    concentration_pmf: NDArray[np.float64],
    time_derivative_concentration_pmf: NDArray[np.float64],
    chamber_parameters: ChamberParameters,
    fit_guess: NDArray[np.float64],
    fit_bounds: List[Tuple[float, float]],
    minimize_method: str = "L-BFGS-B",
) -> Tuple[float, float]: ...
```

#### See also

- [ChamberParameters](#chamberparameters)



## optimize_parameters

[Show source in chamber_rate_fitting.py:215](https://github.com/uncscode/particula/blob/main/particula/data/process/chamber_rate_fitting.py#L215)

Get the optimized parameters using the given cost function.

#### Signature

```python
def optimize_parameters(
    cost_function: ignore,
    initial_guess: NDArray[np.float64],
    bounds: List[Tuple[float, float]],
    method: str,
) -> Tuple[float, float]: ...
```
