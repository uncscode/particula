# particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc

*module*

Coagulation strategy module.

Defines an abstract base class and supporting methods for particle
coagulation processes in aerosol simulations.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L1-L1)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L28-L28)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC

*class*

```python
class CoagulationStrategyABC(ABC)
```

Bases: `ABC`

Abstract base class for defining a coagulation strategy.

This class defines the methods that must be implemented by any
coagulation strategy (e.g., for discrete, continuous, or
particle-resolved distributions).

**Attributes**

- `distribution_type` — The type of distribution to be used, one of ("discrete", "continuous_pdf", or "particle_resolved").
- `use_direct_kernel` — Whether to compute pairwise kernels directly for particle-resolved coagulation.

Methods:
- dimensionless_kernel : Calculate the dimensionless coagulation kernel.
- kernel : Obtain the dimensioned coagulation kernel [m^3/s].
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

**Examples**

```py
class ExampleCoagulation(CoagulationStrategyABC):
    def dimensionless_kernel(self, diff_kn, coulomb_phi):
        return diff_kn + coulomb_phi
    def kernel(self, particle, temperature, pressure):
        return 1.0
strategy = ExampleCoagulation("discrete")
```

> **References**
>
> Seinfeld, J. H. & Pandis, S. N. (2016). Atmospheric Chemistry and
> Physics: From Air Pollution to Climate Change (3rd ed.). Wiley.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L220-L692)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = distribution_type
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L293-L293)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.particle_resolved_bins_number

*attribute* · *instance attribute*

```python
particle_resolved_bins_number = particle_resolved_kernel_bins_number
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L297-L299)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.particle_resolved_bins_per_decade

*attribute* · *instance attribute*

```python
particle_resolved_bins_per_decade = particle_resolved_kernel_bins_per_decade
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L300-L302)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.particle_resolved_radius

*attribute* · *instance attribute*

```python
particle_resolved_radius = particle_resolved_kernel_radius
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L296-L296)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.random_generator

*attribute* · *instance attribute*

```python
random_generator = np.random.default_rng()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L294-L294)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.use_direct_kernel

*attribute* · *instance attribute*

```python
use_direct_kernel = use_direct_kernel
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L303-L303)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.__init__

*method*

```python
def __init__(distribution_type: str, particle_resolved_kernel_radius: Optional[NDArray[np.float64]] = None, particle_resolved_kernel_bins_number: Optional[int] = None, particle_resolved_kernel_bins_per_decade: int = 10, use_direct_kernel: bool = False)
```

Initialize the coagulation strategy.

**Parameters**

- `distribution_type` (`str`) — Type of distribution ("discrete", "continuous_pdf", or "particle_resolved").
- `particle_resolved_kernel_radius` (`Optional[NDArray[np.float64]]`) (default: `None`) — Kernel radius for particle-resolved simulations.
- `particle_resolved_kernel_bins_number` (`Optional[int]`) (default: `None`) — Number of bins for particle-resolved kernel.
- `particle_resolved_kernel_bins_per_decade` (`int`) (default: `10`) — Bins per decade for particle-resolved kernel (default 10).
- `use_direct_kernel` (`bool`) (default: `False`) — Whether to compute kernel values directly for particle-resolved coagulation instead of using interpolation.

**Raises**

- `ValueError` — If distribution_type is not valid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L259-L303)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.coulomb_potential_ratio

*method*

```python
def coulomb_potential_ratio(particle: ParticleData, temperature: float) -> NDArray[np.float64]
```

Calculate the Coulomb potential ratio for each particle.

This ratio characterizes the influence of electrostatic forces on
coagulation processes. CPU ``ParticleData`` inputs are supported only
for ``n_boxes == 1``.

**Parameters**

- `particle` (`ParticleData`) — The particle data.
- `temperature` (`float`) — The gas-phase temperature [K].

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Coulomb potential ratio(s) [dimensionless].

**Raises**

- `ValueError` — If a CPU ``ParticleData`` input has ``n_boxes != 1``.

**Examples**

```py
phi = strategy.coulomb_potential_ratio(particle, 298.15)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L601-L636)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.diffusive_knudsen

*method*

```python
def diffusive_knudsen(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Calculate the diffusive Knudsen number for each particle.

The Knudsen number is used to characterize the relative importance of
diffusion-controlled processes. CPU ``ParticleData`` inputs are
supported only for ``n_boxes == 1``.

**Parameters**

- `particle` (`ParticleData`) — The particle data.
- `temperature` (`float`) — The gas-phase temperature [K].
- `pressure` (`float`) — The gas-phase pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Diffusive Knudsen number(s) [dimensionless].

**Raises**

- `ValueError` — If a CPU ``ParticleData`` input has ``n_boxes != 1``.

**Examples**

```py
knudsen_nums = strategy.diffusive_knudsen(particle, 298.15, 101325)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L550-L599)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.dimensionless_kernel

*method* · *abstract*

```python
def dimensionless_kernel(diffusive_knudsen: NDArray[np.float64], coulomb_potential_ratio: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the dimensionless coagulation kernel.

**Parameters**

- `diffusive_knudsen` (`NDArray[np.float64]`) — The diffusive Knudsen number [dimensionless].
- `coulomb_potential_ratio` (`NDArray[np.float64]`) — The Coulomb potential ratio [dimensionless].

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Dimensionless kernel for particle coagulation.

**Examples**

```py
H = strategy.dimensionless_kernel(kn_array, phi_array)
# H might be array([...]) representing the dimensionless kernel
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L305-L328)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.friction_factor

*method*

```python
def friction_factor(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute the friction factor for each particle in the aerosol.

Considers dynamic viscosity, mean free path, and slip correction to
determine the friction factor [kg/s]. CPU ``ParticleData``
inputs are supported only for ``n_boxes == 1``.

**Parameters**

- `particle` (`ParticleData`) — The particle data for which to compute friction factor.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Friction factor(s) [kg/s].

**Raises**

- `ValueError` — If a CPU ``ParticleData`` input has ``n_boxes != 1``.

**Examples**

```py
fr = strategy.friction_factor(particle, 298.15, 101325)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L638-L692)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.gain_rate

*method*

```python
def gain_rate(particle: ParticleData, kernel: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]
```

Calculate the coagulation gain rate.

Rates use 1/m^3/s for PMF and 1/m^4/s for radius PDF.

CPU ``ParticleData`` inputs are supported only for ``n_boxes == 1``.

**Parameters**

- `particle` (`ParticleData`) — The particle data used in the calculation.
- `kernel` (`NDArray[np.float64]`) — The coagulation kernel [m^3/s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The gain rate in 1/m^3/s for PMF or 1/m^4/s for radius PDF.

**Raises**

- `ValueError` — If the distribution type is invalid.
- `ValueError` — If a CPU ``ParticleData`` input has ``n_boxes != 1``.

**Examples**

```py
gain = strategy.gain_rate(particle, k_val)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L409-L468)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.kernel

*method* · *abstract*

```python
def kernel(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Calculate the coagulation kernel [m^3/s].

Uses particle attributes (e.g., radius, mass) along with temperature
and pressure to return a dimensional kernel for coagulation. CPU
``ParticleData`` inputs are supported only for ``n_boxes == 1``.

**Parameters**

- `particle` (`ParticleData`) — Explicitly tagged ParticleData providing radius and concentration.
- `temperature` (`float`) — The temperature in Kelvin [K].
- `pressure` (`float`) — The pressure in Pascals [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — float or NDArray[np.float64]: The coagulation kernel [m^3/s].

**Raises**

- `ValueError` — If a CPU ``ParticleData`` input has ``n_boxes != 1``.

**Examples**

```py
k_val = strategy.kernel(particle, 298.15, 101325)
# k_val can be a scalar or array
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L330-L360)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, kernel: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]
```

Calculate the coagulation loss rate.

Rates use 1/m^3/s for PMF and 1/m^4/s for radius PDF.

CPU ``ParticleData`` inputs are supported only for ``n_boxes == 1``.

**Parameters**

- `particle` (`ParticleData`) — The particle data for which the loss rate is calculated.
- `kernel` (`NDArray[np.float64]`) — The coagulation kernel [m^3/s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The loss rate in 1/m^3/s for PMF or 1/m^4/s for radius PDF.

**Raises**

- `ValueError` — If the distribution type is invalid.
- `ValueError` — If a CPU ``ParticleData`` input has ``n_boxes != 1``.

**Examples**

```py
loss = strategy.loss_rate(particle, k_val)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L362-L407)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.net_rate

*method*

```python
def net_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Compute the net coagulation rate as gain minus loss.

Rates use 1/m^3/s for PMF and 1/m^4/s for radius PDF.

CPU ``ParticleData`` inputs are supported only for ``n_boxes == 1``.

**Parameters**

- `particle` (`ParticleData`) — The particle data.
- `temperature` (`float`) — The gas-phase temperature [K].
- `pressure` (`float`) — The gas-phase pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — float or NDArray[np.float64] : The net coagulation rate
- (`Union[float, NDArray[np.float64]]`) — in 1/m^3/s for PMF or 1/m^4/s for radius PDF.
- (`Union[float, NDArray[np.float64]]`) — Positive values mean net gain; negative values mean net loss.

**Raises**

- `ValueError` — If a CPU ``ParticleData`` input has ``n_boxes != 1``.

**Examples**

```py
net = strategy.net_rate(particle, 298.15, 101325)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L470-L508)

### particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Perform a single coagulation step over a specified time interval.

Update density populations from their net rate or merge selected
resolved collisions. Explicit distribution metadata and a single box
are required, including on zero-duration calls.

**Parameters**

- `particle` (`ParticleData`) — The particle data to update.
- `temperature` (`float`) — The gas-phase temperature [K].
- `pressure` (`float`) — The gas-phase pressure [Pa].
- `time_step` (`float`) — The timestep over which to integrate [s].

**Returns**

- `ParticleData` (`ParticleData`) — The identical updated native container.

**Raises**

- `TypeError` — If particle is not ParticleData.
- `ValueError` — If distribution metadata is absent or unsupported,
or if a CPU ParticleData input has ``n_boxes != 1``.

**Examples**

```py
updated_particle = strategy.step(particle, 298.15, 101325, 1.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L510-L548)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._build_direct_kernel_index_func

*function*

```python
def _build_direct_kernel_index_func(particle_radius: NDArray[np.float64], particle_mass: NDArray[np.float64], particle_charge: NDArray[np.float64], kernel_builder: Callable[..., NDArray[np.float64]], temperature: float, pressure: float) -> Callable[[NDArray[np.int64], NDArray[np.int64]], NDArray[np.float64]]
```

Build an index-based kernel function for particle-resolved sampling.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L166-L217)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_binned_kernel_data_from_particle_data

*function*

```python
def _get_binned_kernel_data_from_particle_data(data: ParticleData, kernel_radius: NDArray[np.float64]) -> ParticleData
```

Bin particle-resolved ParticleData into a kernel-friendly container.

**Parameters**

- `data` (`ParticleData`) — ParticleData to bin.
- `kernel_radius` (`NDArray[np.float64]`) — Radius bin edges for kernel calculations.

**Returns**

- (`ParticleData`) — ParticleData with binned masses, concentration, and charge.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L120-L133)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_charge

*function*

```python
def _get_charge(particle: ParticleData) -> NDArray[np.float64]
```

Return explicit material charge lanes.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L73-L75)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_concentration

*function*

```python
def _get_concentration(particle: ParticleData) -> NDArray[np.float64]
```

Interpret explicitly tagged population state as slot number density.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L66-L70)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_effective_density

*function*

```python
def _get_effective_density(particle: ParticleData) -> NDArray[np.float64]
```

Return per-particle effective material density.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L83-L85)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_mass

*function*

```python
def _get_mass(particle: ParticleData) -> NDArray[np.float64]
```

Return total per-particle material mass.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L61-L63)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_mean_effective_density

*function*

```python
def _get_mean_effective_density(particle: ParticleData) -> float
```

Return arithmetic mean effective density excluding zero-mass slots.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L88-L94)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_particle_resolved_kernel_radius_data

*function*

```python
def _get_particle_resolved_kernel_radius_data(data: ParticleData, bin_radius: Optional[NDArray[np.float64]] = None, total_bins: Optional[int] = None, bins_per_radius_decade: int = 10) -> NDArray[np.float64]
```

Determine binned radii for ParticleData kernel calculations.

**Parameters**

- `data` (`ParticleData`) — ParticleData input for radius binning.
- `bin_radius` (`Optional[NDArray[np.float64]]`) (default: `None`) — Optional array of radius bin edges in meters.
- `total_bins` (`Optional[int]`) (default: `None`) — Exact number of bins to generate, if set.
- `bins_per_radius_decade` (`int`) (default: `10`) — Number of bins per decade of radius.

**Returns**

- (`NDArray[np.float64]`) — Bin edges (radii) in meters.

**Raises**

- `ValueError` — If finite radii cannot be determined for binning.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L97-L117)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_radius

*function*

```python
def _get_radius(particle: ParticleData) -> NDArray[np.float64]
```

Return material radii from single-box native storage.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L56-L58)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._get_volume

*function*

```python
def _get_volume(particle: ParticleData) -> float
```

Return the explicit single-box volume.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L78-L80)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._require_single_box_particle_data

*function*

```python
def _require_single_box_particle_data(particle: ParticleData) -> ParticleData
```

Validate that ParticleData uses the supported single-box CPU layout.

**Parameters**

- `particle` (`ParticleData`) — ParticleData input to validate.

**Returns**

- (`ParticleData`) — The original ParticleData instance when ``n_boxes == 1``.

**Raises**

- `ValueError` — If ``particle.n_boxes`` is not 1.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L31-L53)

## particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc._resolve_radius_indices

*function*

```python
def _resolve_radius_indices(particle_radius: NDArray[np.float64], lookup_values: NDArray[np.float64]) -> NDArray[np.int64]
```

Resolve radius values to unique particle indices.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Available particle radii.
- `lookup_values` (`NDArray[np.float64]`) — Radii to resolve against particle_radius.

**Returns**

- (`NDArray[np.int64]`) — Index array aligned with lookup_values.

**Raises**

- `ValueError` — If a lookup value has zero or multiple matches.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L136-L163)
