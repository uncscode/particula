# Particle Resolved Method

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Particle Resolved Method

> Auto-generated documentation for [particula.next.dynamics.coagulation.particle_resolved_method](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py) module.

## calculate_probabilities

[Show source in particle_resolved_method.py:33](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py#L33)

Calculate coagulation probabilities based on kernel values and system
parameters.

#### Arguments

- `kernel_values` *float* - Interpolated kernel value for a particle pair.
- `time_step` *float* - The time step over which coagulation occurs.
- `events` *int* - Number of possible coagulation events.
- `tests` *int* - Number of tests (or trials) for coagulation.
- `volume` *float* - Volume of the system.

#### Returns

- `float` - Coagulation probability.

#### Signature

```python
def calculate_probabilities(
    kernel_values: Union[float, NDArray[np.float64]],
    time_step: float,
    events: int,
    tests: int,
    volume: float,
) -> Union[float, NDArray[np.float64]]: ...
```



## interpolate_kernel

[Show source in particle_resolved_method.py:15](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py#L15)

Create a 2D interpolation function for the coagulation kernel.

#### Arguments

- `kernel` *NDArray[np.float64]* - Coagulation kernel.
- `kernel_radius` *NDArray[np.float64]* - Radii corresponding to kernel
    bins.

#### Returns

- `RectBivariateSpline` - Interpolated kernel function.

#### Signature

```python
def interpolate_kernel(
    kernel: NDArray[np.float64], kernel_radius: NDArray[np.float64]
) -> RectBivariateSpline: ...
```



## particle_resolved_coagulation_step

[Show source in particle_resolved_method.py:140](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py#L140)

Perform a single step of particle coagulation, updating particle radii
based on coagulation events.

#### Arguments

- `particle_radius` *NDArray[np.float64]* - Array of particle radii.
- `kernel` *NDArray[np.float64]* - Coagulation kernel as a 2D array where
    each element represents the probability of coagulation between
    particles of corresponding sizes.
- `kernel_radius` *NDArray[np.float64]* - Array of radii corresponding to
    the kernel bins.
- `volume` *float* - Volume of the system in which coagulation occurs.
- `time_step` *float* - Time step over which coagulation is calculated.
- `random_generator` *np.random.Generator* - Random number generator for
    stochastic processes.

#### Returns

- `NDArray[np.int64]` - Array of indices corresponding to the coagulation
    events, where each element is a pair of indices corresponding to
    the coagulating particles [loss, gain].

#### Signature

```python
def particle_resolved_coagulation_step(
    particle_radius: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> NDArray[np.int64]: ...
```



## particle_resolved_update_step

[Show source in particle_resolved_method.py:94](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py#L94)

Update the particle radii and concentrations after coagulation events.

#### Arguments

- `particle_radius` *NDArray[float64]* - Array of particle radii.
- `small_index` *NDArray[int64]* - Indices corresponding to smaller
    particles.
- `large_index` *NDArray[int64]* - Indices corresponding to larger
    particles.

#### Returns

- Updated array of particle radii.
- Updated array for the radii of particles that were lost.
- Updated array for the radii of particles that were gained.

#### Signature

```python
def particle_resolved_update_step(
    particle_radius: NDArray[np.float64],
    loss: NDArray[np.float64],
    gain: NDArray[np.float64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
```



## resolve_final_coagulation_state

[Show source in particle_resolved_method.py:57](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/particle_resolved_method.py#L57)

Resolve the final state of particles that have undergone multiple
coagulation events.

#### Arguments

- `small_indices` *NDArray[np.int64]* - Indices of smaller particles.
- `large_indices` *NDArray[np.int64]* - Indices of larger particles.
- `particle_radius` *NDArray[np.float64]* - Radii of particles.

#### Returns

- `Tuple[NDArray[np.int64],` *NDArray[np.int64]]* - Updated small and large
indices.

#### Signature

```python
def resolve_final_coagulation_state(
    small_indices: NDArray[np.int64],
    large_indices: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```
