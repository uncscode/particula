# Super Droplet Method

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Super Droplet Method

> Auto-generated documentation for [particula.next.dynamics.coagulation.super_droplet_method](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py) module.

## bin_particles

[Show source in super_droplet_method.py:449](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L449)

Bin particles by size and return the number of particles in each bin.

#### Arguments

- `particle_radius` - Array of sorted particle radii.
- `radius_bins` - Array defining the bin edges for particle radii.

#### Returns

Tuple:
    - Array of the number of particles in each bin.
    - Array of bin indices for each particle.

#### Signature

```python
def bin_particles(
    particle_radius: NDArray[np.float64], radius_bins: NDArray[np.float64]
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## bin_to_particle_indices

[Show source in super_droplet_method.py:275](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L275)

Convert bin indices to actual particle indices in the particle array.

This function calculates the actual indices in the particle array
corresponding to the bins specified by `lower_bin` and `upper_bin`.
The function adjusts the provided bin-relative indices to reflect
their position in the full particle array.

#### Arguments

- `lower_indices` - Array of indices relative to the start of
    the `lower_bin`.
- `upper_indices` - Array of indices relative to the start of
    the `upper_bin`.
- `lower_bin` - Index of the bin containing smaller particles.
- `upper_bin` - Index of the bin containing larger particles.
- `bin_indices` - Array containing the start indices of each bin in the
    particle array.

#### Returns

Tuple:
    - `-` *`small_index`* - Indices of particles from the `lower_bin`.
    - `-` *`large_index`* - Indices of particles from the `upper_bin`.

#### Signature

```python
def bin_to_particle_indices(
    lower_indices: NDArray[np.int64],
    upper_indices: NDArray[np.int64],
    lower_bin: int,
    upper_bin: int,
    bin_indices: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## calculate_concentration_in_bins

[Show source in super_droplet_method.py:492](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L492)

Calculate the concentration of particles in each bin.

#### Arguments

- `bin_indices` - Array of bin indices for each particle.
- `particle_concentration` - Array of sorted particle concentrations.
number_in_bins : Array of the number of particles in each bin.

#### Returns

The total concentration in each bin.

#### Signature

```python
def calculate_concentration_in_bins(
    bin_indices: NDArray[np.int64],
    particle_concentration: NDArray[np.float64],
    number_in_bins: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## coagulation_events

[Show source in super_droplet_method.py:363](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L363)

Calculate coagulation probabilities and filter events based on them.

This function calculates the probability of coagulation events occurring
between pairs of particles, based on the ratio of the kernel value for
each pair to the maximum kernel value for the bins. The function then
randomly determines which events occur using these probabilities.

#### Arguments

- `small_index` - Array of indices for the first set of particles
    (smaller particles) involved in the events.
- `large_index` - Array of indices for the second set of particles
    (larger particles) involved in the events.
- `kernel_values` - Array of kernel values corresponding to the
    particle pairs.
- `kernel_max` - The maximum kernel value used for normalization
    of probabilities.
- `generator` - A NumPy random generator used to sample random numbers.

#### Returns

Tuple:
    - Filtered `small_index` array containing indices where
        coagulation events occurred.
    - Filtered `large_index` array containing indices where
        coagulation events occurred.

#### Signature

```python
def coagulation_events(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    kernel_values: NDArray[np.float64],
    kernel_max: float,
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## event_pairs

[Show source in super_droplet_method.py:102](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L102)

Calculate the number of particle pairs based on kernel value.

#### Arguments

- `lower_bin` - Lower bin index.
- `upper_bin` - Upper bin index.
- `kernel_max` - Maximum value of the kernel.
- `number_in_bins` - Number of particles in each bin.

#### Returns

The number of particle pairs events based on the kernel and
number of particles in the bins.

#### Signature

```python
def event_pairs(
    lower_bin: int,
    upper_bin: int,
    kernel_max: Union[float, NDArray[np.float64]],
    number_in_bins: Union[NDArray[np.float64], NDArray[np.int64]],
) -> float: ...
```



## filter_valid_indices

[Show source in super_droplet_method.py:317](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L317)

Filter particles indices based on particle radius and event counters.

This function filters out particle indices that are considered invalid
based on two criteria:
1. The particle radius must be greater than zero.
2. If provided, the single event counter must be less than one.

#### Arguments

- `small_index` - Array of indices for particles in the smaller bin.
- `large_index` - Array of indices for particles in the larger bin.
- `particle_radius` - Array containing the radii of particles.
- `single_event_counter` *Optional* - Optional array tracking the
    number of events for each particle. If provided, only particles
    with a counter value less than one are valid.

#### Returns

Tuple:
    - Filtered `small_index` array containing only valid indices.
    - Filtered `large_index` array containing only valid indices.

#### Signature

```python
def filter_valid_indices(
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
    particle_radius: NDArray[np.float64],
    single_event_counter: Optional[NDArray[np.int64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## get_bin_pairs

[Show source in super_droplet_method.py:476](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L476)

Pre-compute the unique bin pairs for vectorized operations.

#### Arguments

- `bin_indices` - Array of bin indices.

#### Returns

Unique bin pairs for vectorized operations.

#### Signature

```python
def get_bin_pairs(bin_indices: NDArray[np.int64]) -> list[Tuple[int, int]]: ...
```



## random_choice_indices

[Show source in super_droplet_method.py:165](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L165)

Filter valid indices and select random indices for coagulation events.

This function filters particle indices based on bin indices and ensures
the selected particles have a positive radius. It then randomly selects
indices from both a lower bin and an upper bin for a given number of
events.

#### Arguments

- `lower_bin` - The index of the lower bin to filter particles from.
- `upper_bin` - The index of the upper bin to filter particles from.
- `events` - Number of events (indices) to sample for each bin.
- `particle_radius` - A NumPy array of particle radii. Only particles with
    radius > 0 are considered.
- `bin_indices` - A NumPy array of bin indices corresponding to each
    particle.
- `generator` - A NumPy random generator used to sample indices.

#### Returns

Tuple:
    - Indices of particles from the lower bin.
    - Indices of particles from the upper bin.

#### Examples

``` py title="Example choice indices (update)"
rng = np.random.default_rng()
particle_radius = np.array([0.5, 0.0, 1.2, 0.3, 0.9])
bin_indices = np.array([1, 1, 1, 2, 2])
lower_bin = 1
upper_bin = 2
events = 2
lower_indices, upper_indices = random_choice_indices(
    lower_bin, upper_bin, events, particle_radius, bin_indices, rng)
# lower_indices: array([0, 4])
# upper_indices: array([0, 1])
```

#### Signature

```python
def random_choice_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    particle_radius: NDArray[np.float64],
    bin_indices: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## sample_events

[Show source in super_droplet_method.py:133](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L133)

Sample the number of coagulation events from a Poisson distribution.

This function calculates the expected number of coagulation events based on
the number of particle pairs, the simulation volume, and the time step. It
then samples the actual number of events using a Poisson distribution.

#### Arguments

- `events` - The calculated number of particle pairs that could
    interact.
- `volume` - The volume of the simulation space.
- `time_step` - The time step over which the events are being simulated.
- `generator` - A NumPy random generator used to sample from the Poisson
    distribution.

#### Returns

The sampled number of coagulation events as an integer.

#### Signature

```python
def sample_events(
    events: float, volume: float, time_step: float, generator: np.random.Generator
) -> int: ...
```



## select_random_indices

[Show source in super_droplet_method.py:228](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L228)

Select random indices for particles involved in coagulation events.

This function generates random indices for particles in the specified bins
(`lower_bin` and `upper_bin`) that are involved in a specified number of
events. The indices are selected based on the number of particles in
each bin.

#### Arguments

- `lower_bin` - Index of the bin containing smaller particles.
- `upper_bin` - Index of the bin containing larger particles.
- `events` - The number of events to sample indices for.
- `number_in_bins` - Array representing the number of particles in
    each bin.
- `generator` - A NumPy random generator used to sample indices.

#### Returns

Tuple:
    - Indices of particles from `lower_bin`.
    - Indices of particles from `upper_bin`.

#### Signature

```python
def select_random_indices(
    lower_bin: int,
    upper_bin: int,
    events: int,
    number_in_bins: NDArray[np.int64],
    generator: np.random.Generator,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]: ...
```



## sort_particles

[Show source in super_droplet_method.py:411](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L411)

Sort particles by size and optionally sort their concentrations.

#### Arguments

- `particle_radius` - Array of particle radii.
- `particle_concentration` - Optional array of particle concentrations
    corresponding to each radius. If provided, it will be sorted to
    match the sorted radii.

#### Returns

Tuple:
    - `-` *`unsort_indices`* - Array of indices to revert the sorting.
    - `-` *`sorted_radius`* - Array of sorted particle radii.
    - `-` *`sorted_concentration`* - Optional array of sorted particle
        concentrations (or `None` if not provided).

#### Signature

```python
def sort_particles(
    particle_radius: NDArray[np.float64],
    particle_concentration: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.float64], Optional[NDArray[np.float64]]]: ...
```



## super_droplet_coagulation_step

[Show source in super_droplet_method.py:520](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L520)

Perform a single step of the Super Droplet coagulation process.

This function processes particles by sorting them, binning by size,
computing coagulation events based on the coagulation kernel, and
updating particle properties accordingly.

#### Arguments

- `particle_radius` - Array of particle radii.
- `particle_concentration` - Array of particle concentrations
    corresponding to each radius.
- `kernel` - 2D array representing the coagulation kernel values between
    different bins.
- `kernel_radius` - Array defining the radii corresponding to the
    kernel bins.
- `volume` - Volume of the system or relevant scaling factor.
- `time_step` - Duration of the current time step.
random_generator : A NumPy random number generator for
    stochastic processes.

#### Returns

Tuple:
    - Updated array of particle radii after coagulation.
    - Updated array of particle concentrations after coagulation.

#### Signature

```python
def super_droplet_coagulation_step(
    particle_radius: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
    kernel: NDArray[np.float64],
    kernel_radius: NDArray[np.float64],
    volume: float,
    time_step: float,
    random_generator: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...
```



## super_droplet_update_step

[Show source in super_droplet_method.py:14](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/super_droplet_method.py#L14)

Update the particle radii and concentrations after coagulation events.

#### Arguments

- `particle_radius` *NDArray[float64]* - Array of particle radii.
- `concentration` *NDArray[float64]* - Array representing the concentration
    of particles.
- `single_event_counter` *NDArray[int64]* - Tracks the number of
    coagulation events for each particle.
- `small_index` *NDArray[int64]* - Indices corresponding to smaller
    particles.
- `large_index` *NDArray[int64]* - Indices corresponding to larger
    particles.

#### Returns

- Updated array of particle radii.
- Updated array representing the concentration of particles.
- Updated array tracking the number of coagulation events.

#### Signature

```python
def super_droplet_update_step(
    particle_radius: NDArray[np.float64],
    concentration: NDArray[np.float64],
    single_event_counter: NDArray[np.int64],
    small_index: NDArray[np.int64],
    large_index: NDArray[np.int64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]: ...
```
