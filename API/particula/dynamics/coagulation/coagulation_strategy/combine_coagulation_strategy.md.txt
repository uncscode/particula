# particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy

*module*

Combine multiple coagulation strategies into a single, aggregated approach.

The kernel from each strategy is summed to form a unified coagulation
kernel. All strategies must share the same distribution type.

**Classes**

- `- CombineCoagulationStrategy ` — Inherits from CoagulationStrategyABC
and aggregates multiple strategies.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L1-L1)

## particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L23-L23)

## particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy

*class*

```python
class CombineCoagulationStrategy(CoagulationStrategyABC)
```

Bases: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

Combine multiple coagulation strategies into one.

This class takes a list of coagulation strategies and merges their
kernels by summing them. Each included strategy must share the same
distribution type.

**Attributes**

- `distribution_type` — Matches the distribution_type of the first strategy.
- `strategies` — A list of individual CoagulationStrategyABC instances.

Methods:
- dimensionless_kernel : Raises NotImplementedError, as not supported here.
- kernel : Compute the sum of all strategy kernels.
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

**Examples**

```py title="Example Usage of CombineCoagulationStrategy"
import particula as par
combined_strategy = par.dynamics.CombineCoagulationStrategy(
    [strategy1, strategy2]
)
k_value = combined_strategy.kernel(
    particle=aerosol.particles, temperature=300, pressure=101325
) # combined kernel value
```

> **References**
>
> - No specific references. Summation approach is straightforward.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L26-L158)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.strategies

*attribute* · *instance attribute*

```python
strategies = strategies
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L91-L91)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = distribution_type
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L293-L293)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.particle_resolved_bins_number

*attribute* · *instance attribute*

```python
particle_resolved_bins_number = particle_resolved_kernel_bins_number
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L297-L299)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.particle_resolved_bins_per_decade

*attribute* · *instance attribute*

```python
particle_resolved_bins_per_decade = particle_resolved_kernel_bins_per_decade
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L300-L302)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.particle_resolved_radius

*attribute* · *instance attribute*

```python
particle_resolved_radius = particle_resolved_kernel_radius
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L296-L296)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L294-L294)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.use_direct_kernel

*attribute* · *instance attribute*

```python
use_direct_kernel = use_direct_kernel
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L303-L303)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.__init__

*method*

```python
def __init__(strategies: List[CoagulationStrategyABC])
```

Initialize the combined coagulation strategy.

**Parameters**

- `strategies` (`List[CoagulationStrategyABC]`) — A list of CoagulationStrategyABC instances to combine.

**Returns**

- `result` (`None`) — No return value.

**Raises**

- `ValueError` — If no strategies are provided or if they have
mismatched distribution types.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L66-L91)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.dimensionless_kernel

*method*

```python
def dimensionless_kernel(diffusive_knudsen: NDArray[np.float64], coulomb_potential_ratio: NDArray[np.float64]) -> NDArray[np.float64]
```

Raise NotImplementedError for dimensionless kernel in combined
strategy.

Dimensionless kernels must be handled individually by the underlying
strategies. This method logs an error and raises NotImplementedError.

**Parameters**

- `diffusive_knudsen` (`NDArray[np.float64]`) — The diffusive Knudsen number(s) [dimensionless].
- `coulomb_potential_ratio` (`NDArray[np.float64]`) — The Coulomb potential ratio(s) [dimensionless].

**Raises**

- `NotImplementedError` — This method is not supported in the
combined strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L93-L119)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.kernel

*method*

```python
def kernel(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Compute the total coagulation kernel by summing the kernels from all
underlying strategies.

**Parameters**

- `particle` (`ParticleData`) — The particle data instance containing particle data (radii, distribution, etc.).
- `temperature` (`float`) — The temperature in Kelvin [K].
- `pressure` (`float`) — The pressure in Pascals [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — float or NDArray[np.float64]: The combined coagulation kernel, equal to the sum of each strategy's kernel.

**Examples**

```py
k_combined = combined_strategy.kernel(
    particle=my_particle,
    temperature=300.0,
    pressure=101325
)
# k_combined is the sum of kernels from each strategy in
# combined_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/combine_coagulation_strategy.py#L121-L158)

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.coulomb_potential_ratio

*method*

```python
def coulomb_potential_ratio(particle: ParticleData, temperature: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

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

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.diffusive_knudsen

*method*

```python
def diffusive_knudsen(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

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

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.friction_factor

*method*

```python
def friction_factor(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

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

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.gain_rate

*method*

```python
def gain_rate(particle: ParticleData, kernel: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

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

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, kernel: NDArray[np.float64]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

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

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.net_rate

*method*

```python
def net_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

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

### particula.dynamics.coagulation.coagulation_strategy.combine_coagulation_strategy.CombineCoagulationStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

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
