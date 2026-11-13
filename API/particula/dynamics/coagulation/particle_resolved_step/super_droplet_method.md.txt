# particula.dynamics.coagulation.particle_resolved_step.super_droplet_method

*module*

Super droplet method for coagulation dynamics.

This module implements a Super Droplet Method for coagulation
dynamics, used to simulate how particles grow through collisions.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L1-L1)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._bin_particles

*function*

```python
def _bin_particles(particle_radius: NDArray[np.float64], radius_bins: NDArray[np.float64]) -> Tuple[NDArray[np.int64], NDArray[np.int64]]
```

Divide the sorted particle radii into bins and count how many fall into
each bin.

This function uses `radius_bins` as edges and assigns each particle
radius to a bin index via `np.digitize`. The result is (1) a histogram
with the number of particles in each bin, and (2) an array of per-particle
bin indices.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of sorted particle radii.
- `radius_bins` (`NDArray[np.float64]`) — Edges used to define the bins.

**Returns**

- (`NDArray[np.int64]`) — `number_in_bins`: Counts of how many radii lie in each bin.
- (`NDArray[np.int64]`) — `bin_indices`: The bin index assigned to each particle.

**Examples**

```py
import numpy as np
rad = np.array([1e-9, 1.5e-9, 2e-9, 5e-9])
bin_edges = np.array([1e-9, 2e-9, 3e-9, 1e-8])
n_in_bins, bin_idx = _bin_particles(rad, bin_edges)
# n_in_bins -> [1, 2, 1]
# bin_idx might be [0, 1, 1, 2]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L534-L572)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._bin_to_particle_indices

*function*

```python
def _bin_to_particle_indices(lower_indices: NDArray[np.int64], upper_indices: NDArray[np.int64], lower_bin: int, upper_bin: int, bin_indices: NDArray[np.int64]) -> Tuple[NDArray[np.int64], NDArray[np.int64]]
```

Map bin-relative indices back to absolute positions in the particle
array.

This function adjusts the offsets for each bin so that the pairwise
indices used for collision are mapped onto the actual sorted particle
array. For instance, if `lower_indices` are all within bin 0, and bin 0
particles occupy positions [0..9], this method adds that offset to
each index in `lower_indices`.

**Parameters**

- `lower_indices` (`NDArray[np.int64]`) — Relative indices (local to the bin) of smaller particles.
- `upper_indices` (`NDArray[np.int64]`) — Relative indices (local to the bin) of larger particles.
- `lower_bin` (`int`) — The bin representing the smaller particles.
- `upper_bin` (`int`) — The bin representing the larger particles.
- `bin_indices` (`NDArray[np.int64]`) — Cumulative offsets to determine where each bin begins.

**Returns**

- (`NDArray[np.int64]`) — `small_index`: Absolute positions of smaller particles in the sorted particle array.
- (`NDArray[np.int64]`) — `large_index`: Absolute positions of the larger particles in the sorted array.

**Examples**

```py
bins = np.array([0, 10, 20])
lw_rel = np.array([0, 1])
up_rel = np.array([2, 3])
# Convert these local indices for bin 1 (start=10) and bin 2 (start=20)
s_idx, l_idx = _bin_to_particle_indices(lw_rel, up_rel, 1, 2, bins)
# s_idx -> [10, 11]
# l_idx -> [22, 23]

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L321-L371)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._calculate_concentration_in_bins

*function*

```python
def _calculate_concentration_in_bins(bin_indices: NDArray[np.int64], particle_concentration: NDArray[np.float64], number_in_bins: NDArray[np.float64]) -> NDArray[np.float64]
```

Sum the particle concentrations in each bin.

Given per-particle `bin_indices` and `particle_concentration`, this
function accumulates the total concentration of all particles that
fall into each bin. The `number_in_bins` array is used mainly for
shape reference but can also confirm the count of particles.

**Parameters**

- `bin_indices` (`NDArray[np.int64]`) — Array of bin indices for each particle.
- `particle_concentration` (`NDArray[np.float64]`) — 1D array of concentrations matching each particle.
- `number_in_bins` (`NDArray[np.float64]`) — Array with the count of particles in each bin.

**Returns**

- (`NDArray[np.float64]`) — A 1D array whose length is the number of unique bins, containing the summed concentration per bin.

**Examples**

```py
import numpy as np
b_idx = np.array([0, 0, 1, 1, 2])
conc = np.array([10., 5., 2., 3., 4.])
n_in_bins = np.array([2, 2, 1])  # might match the bin partition
bin_c = _calculate_concentration_in_bins(b_idx, conc, n_in_bins)
# bin_c -> [15., 5., 4.]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L604-L644)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._coagulation_events

*function*

```python
def _coagulation_events(small_index: NDArray[np.int64], large_index: NDArray[np.int64], kernel_values: NDArray[np.float64], kernel_max: float, generator: np.random.Generator) -> Tuple[NDArray[np.int64], NDArray[np.int64]]
```

Stochastically pick which collisions (among possible pairs) actually
happen.

This function computes a collision probability for each `(small_index,
large_index)` pair by taking the ratio of `kernel_values / kernel_max`.
Next, a random uniform draw decides if each collision occurs.

**Parameters**

- `small_index` (`NDArray[np.int64]`) — Array of indices representing smaller particles.
- `large_index` (`NDArray[np.int64]`) — Array of indices representing larger particles.
- `kernel_values` (`NDArray[np.float64]`) — Collision kernel values for each pair.
- `kernel_max` (`float`) — A maximum kernel value used for normalization.
- `generator` (`np.random.Generator`) — Random generator to compare probabilities vs. uniform draws.

**Returns**

- (`NDArray[np.int64]`) — Filtered `small_index` containing only those that coagulated.
- (`NDArray[np.int64]`) — Filtered `large_index` containing only those that coagulated.

**Examples**

```py
rng = np.random.default_rng(999)
s_idx = np.array([0, 1, 2])
l_idx = np.array([3, 4, 5])
kv = np.array([0.5, 1.0, 0.1])
kmax = 1.0
s_new, l_new = _coagulation_events(s_idx, l_idx, kv, kmax, rng)
# Each pair has probability kv/kmax => [0.5, 1.0, 0.1]
# The final s_new, l_new depends on random draws
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L432-L482)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._event_pairs

*function*

```python
def _event_pairs(lower_bin: int, upper_bin: int, kernel_max: Union[float, NDArray[np.float64]], number_in_bins: Union[NDArray[np.float64], NDArray[np.int64]]) -> float
```

Calculate an approximate count of particle-pair interactions.

This function estimates the number of collisions or interactions
that might occur between two bins of particles, given a maximum
kernel value and the current population of each bin. When the bins
are the same, a correction factor is applied to avoid double-counting
pairs.

**Parameters**

- `lower_bin` (`int`) — Index of the lower bin in the distribution.
- `upper_bin` (`int`) — Index of the upper bin in the distribution.
- `kernel_max` (`Union[float, NDArray[np.float64]]`) — Maximum kernel value used to weight collisions.
- `number_in_bins` (`Union[NDArray[np.float64], NDArray[np.int64]]`) — The population of particles per bin.

**Returns**

- (`float`) — A float representing the expected number of particle-pair collision events.

**Examples**

```py
max_kernel = 1.0e-9
n_bins = np.array([100, 150, 200])
# lower_bin=0, upper_bin=1 => collisions between bin 0 and bin 1
events_est = _event_pairs(0, 1, max_kernel, n_bins)
# events_est is ~ 1.0e-9 * 100 * 150
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L117-L160)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._filter_valid_indices

*function*

```python
def _filter_valid_indices(small_index: NDArray[np.int64], large_index: NDArray[np.int64], particle_radius: NDArray[np.float64], single_event_counter: Optional[NDArray[np.int64]] = None) -> Tuple[NDArray[np.int64], NDArray[np.int64]]
```

Remove invalid pairs of particles based on radius and optional event
limit.

This function checks each pair of `(small_index, large_index)` to ensure
both have radius > 0. If `single_event_counter` is provided, it further
enforces that each particle has had < 1 event so far (or you can
define your own threshold). The pairs failing these checks are removed.

**Parameters**

- `small_index` (`NDArray[np.int64]`) — Indices for the smaller particles in each pair.
- `large_index` (`NDArray[np.int64]`) — Indices for the larger particles in each pair.
- `particle_radius` (`NDArray[np.float64]`) — Array of radii for each particle.
- `single_event_counter` (`Optional[NDArray[np.int64]]`) (default: `None`) — Optional array counting how many events each particle has undergone. If provided, only particles with counter < 1 pass the filter.

**Returns**

- (`NDArray[np.int64]`) — Filtered `small_index` with only valid pairs.
- (`NDArray[np.int64]`) — Filtered `large_index` that corresponds to valid pairs.

**Examples**

```py
r = np.array([0.1, 0.0, 0.08, 0.02])
c = np.array([0, 0, 0, 0])
small_i = np.array([0, 1, 2])
large_i = np.array([3, 0, 1])
# Filter out pairs with radius <= 0 or event_counter >= 1
s_valid, l_valid = _filter_valid_indices(
    small_i, large_i, r, single_event_counter=c
)
# Indices with r>0 remain in s_valid, l_valid
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L374-L429)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._get_bin_pairs

*function*

```python
def _get_bin_pairs(bin_indices: NDArray[np.int64]) -> list[Tuple[int, int]]
```

Produce the list of all unique (binA, binB) pairs using combinations
with replacement.

This function is useful when we want to iterate over all bin pairs
(including binA == binB) for collision computations. The combination
ensures each pair is returned only once.

**Parameters**

- `bin_indices` (`NDArray[np.int64]`) — Array of bin indices for each particle (though only the unique values matter).

**Returns**

- (`list[Tuple[int, int]]`) — A list of (lower_bin, upper_bin) pairs covering all unique bins in `bin_indices`.

**Examples**

```py
bins = np.array([0, 0, 1, 2, 2])
pairs = _get_bin_pairs(bins)
# pairs -> [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L575-L601)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._sample_events

*function*

```python
def _sample_events(events: float, volume: float, time_step: float, generator: np.random.Generator) -> int
```

Determine how many collisions actually occur using a Poisson draw.

This function uses the expected collision count (`events`) and normalizes
by system `volume` to compute an effective collision rate. It then
samples from a Poisson distribution to obtain the actual number of
collisions happening within the current `time_step`.

**Parameters**

- `events` (`float`) — The calculated number of particle pairs that could interact.
- `volume` (`float`) — The volume of the simulation space (m³).
- `time_step` (`float`) — The time span (seconds) over which collisions are considered.
- `generator` (`np.random.Generator`) — A NumPy random Generator to sample the Poisson distribution.

**Returns**

- (`int`) — The sampled number of coagulation events as an integer.

**Examples**

```py
from numpy.random import default_rng
rng = default_rng(42)
collisions = _sample_events(events=5e3, volume=0.1, time_step=0.01,
    generator=rng)
# collisions might be ~ Poisson( 5e3 / 0.1 * 0.01 ) => Poisson(5)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L163-L201)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._select_random_indices

*function*

```python
def _select_random_indices(lower_bin: int, upper_bin: int, events: int, number_in_bins: NDArray[np.int64], generator: np.random.Generator) -> Tuple[NDArray[np.int64], NDArray[np.int64]]
```

Randomly choose indices within each bin to represent collision partners.

This function picks `events` indices from the population of the
`lower_bin` and `upper_bin`, ignoring any radius or event-limit checks
(those may happen later). The result is two arrays of equal size,
each containing random picks within the respective bins.

**Parameters**

- `lower_bin` (`int`) — Index for the "smaller" bin.
- `upper_bin` (`int`) — Index for the "larger" bin.
- `events` (`int`) — How many pairs to select.
- `number_in_bins` (`NDArray[np.int64]`) — Array with the count of particles in each bin.
- `generator` (`np.random.Generator`) — Random number generator to draw the indices.

**Returns**

- (`NDArray[np.int64]`) — An array of size `events` with random picks from `lower_bin`.
- (`NDArray[np.int64]`) — An array of size `events` with random picks from `upper_bin`.

**Examples**

```py
import numpy as np
rng = np.random.default_rng(42)
n_in_bins = np.array([5, 10, 7])
i_lw, i_up = _select_random_indices(
    lower_bin=0,
    upper_bin=2,
    events=3,
    number_in_bins=n_in_bins,
    generator=rng
)
# i_lw -> random indices in [0..4]
# i_up -> random indices in [0..6]

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L262-L318)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._sort_particles

*function*

```python
def _sort_particles(particle_radius: NDArray[np.float64], particle_concentration: Optional[NDArray[np.float64]] = None) -> Tuple[NDArray[np.int64], NDArray[np.float64], Optional[NDArray[np.float64]]]
```

Sort particle radii (and optionally concentrations) in ascending order.

The function returns an array of `unsort_indices` that can be used
to restore the particles to their original order after manipulations.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — 1D NumPy array of particle radii.
- `particle_concentration` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional array of corresponding concentrations.

**Returns**

- (`NDArray[np.int64]`) — `unsort_indices`: Indices to revert sorting to the original order.
- (`NDArray[np.float64]`) — `sorted_radius`: Sorted array of radii in ascending order.
- (`Optional[NDArray[np.float64]]`) — `sorted_concentration`: Sorted array of concentrations, if provided; otherwise `None`.

**Examples**

```py
import numpy as np
r = np.array([0.3, 0.1, 0.5])
c = np.array([10, 30, 20])
u_idx, s_r, s_c = _sort_particles(r, c)
# s_r -> [0.1, 0.3, 0.5]
# s_c -> [30, 10, 20]
# u_idx can be used to get them back in [0.3, 0.1, 0.5] order
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L485-L531)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method._super_droplet_update_step

*function*

```python
def _super_droplet_update_step(particle_radius: NDArray[np.float64], concentration: NDArray[np.float64], single_event_counter: NDArray[np.int64], small_index: NDArray[np.int64], large_index: NDArray[np.int64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]
```

Update particle radii and concentrations when two particles coagulate.

This function merges smaller and larger particles by combining their
volumes and redistributing particle concentrations. The resulting
radii are computed via volume conservation, and an event counter
tracks how many coagulation events each particle has undergone.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii (m).
- `concentration` (`NDArray[np.float64]`) — Array representing the concentration of each particle (number or mass, depending on usage).
- `single_event_counter` (`NDArray[np.int64]`) — Tracks the number of coagulation events each particle has experienced in the current iteration.
- `small_index` (`NDArray[np.int64]`) — Indices for smaller particles in a coagulation event.
- `large_index` (`NDArray[np.int64]`) — Indices for larger particles in a coagulation event.

**Returns**

- (`NDArray[np.float64]`) — An updated array of particle radii (m) following coagulation.
- (`NDArray[np.float64]`) — An updated array representing the concentration of particles.
- (`NDArray[np.int64]`) — An updated array tracking the index-wise number of events.

**Examples**

```py
import numpy as np
r = np.array([1e-9, 2e-9, 3e-9])
conc = np.array([100., 50., 75.])
events = np.zeros_like(r, dtype=int)
s_idx = np.array([0])
l_idx = np.array([2])
out_r, out_c, out_ev = _super_droplet_update_step(
    r, conc, events, s_idx, l_idx)
# out_r[0] is updated via volume combination with out_r[2].
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L15-L114)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method.get_super_droplet_coagulation_step

*function*

```python
def get_super_droplet_coagulation_step(particle_radius: NDArray[np.float64], particle_concentration: NDArray[np.float64], kernel: NDArray[np.float64], kernel_radius: NDArray[np.float64], volume: float, time_step: float, random_generator: np.random.Generator) -> Tuple[NDArray[np.float64], NDArray[np.float64]]
```

Carry out one time-step of super-droplet-based coagulation.

This function sorts particles by radius, bins them, and then stochastically
computes collision events according to the coagulation kernel. It updates
the particle radii/concentrations, then unsorts them back to the original
order.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii (m).
- `particle_concentration` (`NDArray[np.float64]`) — Array of per-particle concentration.
- `kernel` (`NDArray[np.float64]`) — 2D matrix of coagulation kernel values, dimension ~ len(kernel_radius) × len(kernel_radius).
- `kernel_radius` (`NDArray[np.float64]`) — Array of radius points defining the kernel dimension.
- `volume` (`float`) — System volume or domain size in m³.
- `time_step` (`float`) — The length of this coagulation iteration in seconds.
- `random_generator` (`np.random.Generator`) — Random number generator for sampling collisions.

**Returns**

- (`NDArray[np.float64]`) — Updated radii array after processing coagulation.
- (`NDArray[np.float64]`) — Updated concentrations array after processing coagulation.

**Examples**

```py
import numpy as np
from numpy.random import default_rng
radius = np.array([1e-9, 2e-9, 5e-9])
conc = np.array([100., 50., 10.])
ker_vals = np.ones((3,3))
ker_r = np.array([1e-9, 2e-9, 5e-9])
rng = default_rng(42)
r_new, c_new = get_super_droplet_coagulation_step(
    radius, conc, ker_vals, ker_r, 1e-3, 1.0, rng)
# r_new, c_new have updated values after one super droplet
# coagulation step.

> **References**
>
> - E. W. Tedford and L. A. Perugini, "Superdroplet method
>   in cloud microphysics simulations," J. Atmos. Sci., 2020.
> - Seinfeld, J. H., & Pandis, S. N. *Atmospheric Chemistry and Physics*,
>   Wiley, 2016.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L648-L836)

## particula.dynamics.coagulation.particle_resolved_step.super_droplet_method.random_choice_indices

*function*

```python
def random_choice_indices(lower_bin: int, upper_bin: int, events: int, particle_radius: NDArray[np.float64], bin_indices: NDArray[np.int64], generator: np.random.Generator) -> Tuple[NDArray[np.int64], NDArray[np.int64]]
```

Select valid particle indices in two bins for coagulation events.

This function tries to choose `events` valid indices from
`lower_bin` and `upper_bin`, discarding any particles with radius ≤ 0.
It uses the provided random generator to perform the sampling
with replacement if needed.

**Parameters**

- `lower_bin` (`int`) — Index of the lower bin to filter particles from.
- `upper_bin` (`int`) — Index of the upper bin to filter particles from.
- `events` (`int`) — Number of events (indices) to sample for each bin.
- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii; only those > 0 are considered valid.
- `bin_indices` (`NDArray[np.int64]`) — Array of bin labels corresponding to each particle.
- `generator` (`np.random.Generator`) — Random number generator used for index selection.

**Returns**

- (`NDArray[np.int64]`) — Indices of particles from the lower bin.
- (`NDArray[np.int64]`) — Indices of particles from the upper bin.

**Examples**

```py
import numpy as np
rng = np.random.default_rng(123)
radius = np.array([0.3, 0.1, 0.0, 0.5])
bins = np.array([0, 0, 1, 1])
lw_bin, up_bin = random_choice_indices(0, 1, 2, radius, bins, rng)
# lw_bin -> array of valid picks from bin 0
# up_bin -> array of valid picks from bin 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/particle_resolved_step/super_droplet_method.py#L205-L259)
