# particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method

*module*

Particle-resolved coagulation helpers with bounded kernel interpolation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L1-L1)

## particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method._calculate_probabilities

*function*

```python
def _calculate_probabilities(kernel_values: Union[float, NDArray[np.float64]], time_step: float, events: int, tests: int, volume: float) -> Union[float, NDArray[np.float64]]
```

Calculate coagulation probabilities for sampled particle pairs.

**Parameters**

- `kernel_values` (`Union[float, NDArray[np.float64]]`) — Interpolated kernel values for particle pairs.
- `time_step` (`float`) — Duration of the coagulation step in seconds.
- `events` (`int`) — Number of possible collisions for the pair set.
- `tests` (`int`) — Number of collision trials drawn for this pair set.
- `volume` (`float`) — System volume in cubic meters.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Probability (or array of probabilities) that collisions occur during
- (`Union[float, NDArray[np.float64]]`) — the step.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L298-L318)

## particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method._final_coagulation_state

*function*

```python
def _final_coagulation_state(small_indices: NDArray[np.int64], large_indices: NDArray[np.int64], particle_radius: NDArray[np.float64]) -> Tuple[NDArray[np.int64], NDArray[np.int64]]
```

Resolve chained coagulation events to consistent merge pairs.

When a particle appears as both a small and large partner, this function
remaps merges so each small particle maps to one final large particle.

**Parameters**

- `small_indices` (`NDArray[np.int64]`) — Indices of smaller particles in candidate merges.
- `large_indices` (`NDArray[np.int64]`) — Indices of larger particles in candidate merges.
- `particle_radius` (`NDArray[np.float64]`) — Current particle radii used to order remapping.

**Returns**

- (`Tuple[NDArray[np.int64], NDArray[np.int64]]`) — Tuple of ``(small_indices, large_indices)`` with conflicts resolved.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L321-L355)

## particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method._interpolate_kernel

*function*

```python
def _interpolate_kernel(kernel: NDArray[np.float64], kernel_radius: NDArray[np.float64]) -> RegularGridInterpolator
```

Create a kernel interpolator that zeros out-of-bounds radii.

Non-finite kernel entries are replaced with zeros, linear interpolation is
used within the grid, and a zero ``fill_value`` prevents charge-blind
extrapolation outside the tabulated radius range.

**Parameters**

- `kernel` (`NDArray[np.float64]`) — Two-dimensional coagulation kernel values.
- `kernel_radius` (`NDArray[np.float64]`) — Radii that define the kernel grid.

**Returns**

- (`RegularGridInterpolator`) — RegularGridInterpolator that maps ``[r_small, r_large]`` pairs to
- (`RegularGridInterpolator`) — kernel values with safe bounds handling.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L267-L295)

## particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method.get_particle_resolved_coagulation_step

*function*

```python
def get_particle_resolved_coagulation_step(particle_radius: NDArray[np.float64], kernel: NDArray[np.float64], kernel_radius: NDArray[np.float64], volume: float, time_step: float, random_generator: np.random.Generator, kernel_func: Optional[Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]] = None, kernel_index_func: Optional[Callable[[NDArray[np.int64], NDArray[np.int64]], NDArray[np.float64]]] = None) -> NDArray[np.int64]
```

Perform one stochastic coagulation step for a particle population.

Particles are binned by radius, kernel values are interpolated for each
bin pair, and random trials determine which collisions occur. Returned
indices describe which small particles merged into which large particles.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Radii of particles that may coagulate.
- `kernel` (`NDArray[np.float64]`) — Two-dimensional coagulation kernel aligned to ``kernel_radius``.
- `kernel_radius` (`NDArray[np.float64]`) — Radii that define kernel grid points.
- `volume` (`float`) — System volume in cubic meters.
- `time_step` (`float`) — Duration of the coagulation step in seconds.
- `random_generator` (`np.random.Generator`) — NumPy random generator used for sampling.
- `kernel_func` (`Optional[Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]]`) (default: `None`) — Optional callable for direct kernel evaluation. When provided, it is called with ``(r_small, r_large)`` inputs and should return a 1D array of kernel values.
- `kernel_index_func` (`Optional[Callable[[NDArray[np.int64], NDArray[np.int64]], NDArray[np.float64]]]`) (default: `None`) — Optional callable for direct kernel evaluation using particle indices. When provided, it is called with ``(small_index, large_index)`` arrays and should return a 1D array of kernel values. Takes precedence over ``kernel_func``.

**Returns**

- (`NDArray[np.int64]`) — Array of shape ``(n, 2)`` with ``[small_index, large_index]`` pairs
- (`NDArray[np.int64]`) — describing coagulation events.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L57-L264)

## particula.dynamics.coagulation.particle_resolved_step.particle_resolved_method.get_particle_resolved_update_step

*function*

```python
def get_particle_resolved_update_step(particle_radius: NDArray[np.float64], loss: NDArray[np.float64], gain: NDArray[np.float64], small_index: NDArray[np.int64], large_index: NDArray[np.int64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Update particle radii and track losses and gains from coagulation.

Smaller particles indexed by ``small_index`` are removed and their radii
recorded in ``loss``. The corresponding larger particles indexed by
``large_index`` grow to conserve volume, and their original radii are
stored in ``gain``.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Radii of all particles. Modified in place.
- `loss` (`NDArray[np.float64]`) — Array that records radii of particles removed by coagulation.
- `gain` (`NDArray[np.float64]`) — Array that records radii of particles that absorbed others.
- `small_index` (`NDArray[np.int64]`) — Indices of smaller particles in each coagulation pair.
- `large_index` (`NDArray[np.int64]`) — Indices of larger particles in each coagulation pair.

**Returns**

- (`Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`) — Updated particle radii, loss, and gain arrays after coagulation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/particle_resolved_method.py#L15-L52)
