# particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy

*module*

Brownian coagulation strategy class.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L1-L1)

## particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L21-L21)

## particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy

*class*

```python
class BrownianCoagulationStrategy(CoagulationStrategyABC)
```

Bases: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

Discrete Brownian coagulation strategy class for aerosol simulations.

This class implements methods defined in CoagulationStrategyABC
to simulate Brownian coagulation in particle populations. It calculates
coagulation rates via a Brownian kernel that depends on properties such
as temperature, pressure, and particle radius.

**Attributes**

- `distribution_type` — Defines how particles are represented (e.g., "discrete", "continuous_pdf", or "particle_resolved").

Methods:
- kernel : Calculate the Brownian coagulation kernel (dimensioned).
- loss_rate : Calculate the coagulation loss rate (not shown here).
- gain_rate : Calculate the coagulation gain rate (not shown here).
- net_rate : Calculate the net coagulation rate (not shown here).
- dimensionless_kernel : Not implemented, raises NotImplementedError.
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

**Examples**

```py title="Example Usage of BrownianCoagulationStrategy"
import particula as par
brownian_strat = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete"
)
particle_data = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete"
)
kernel_values = brownian_strat.kernel(
    particle_data, temperature=298, pressure=101325
)
```

> **References**
>
> - `get_brownian_kernel_via_system_state`
> - Seinfeld, J. H., & Pandis, S. N. (2016). "Atmospheric chemistry
>   and physics," Section 13, Table 13.1.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L24-L134)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = distribution_type
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L293-L293)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.particle_resolved_bins_number

*attribute* · *instance attribute*

```python
particle_resolved_bins_number = particle_resolved_kernel_bins_number
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L297-L299)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.particle_resolved_bins_per_decade

*attribute* · *instance attribute*

```python
particle_resolved_bins_per_decade = particle_resolved_kernel_bins_per_decade
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L300-L302)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.particle_resolved_radius

*attribute* · *instance attribute*

```python
particle_resolved_radius = particle_resolved_kernel_radius
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L296-L296)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L294-L294)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.use_direct_kernel

*attribute* · *instance attribute*

```python
use_direct_kernel = use_direct_kernel
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L303-L303)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.__init__

*method*

```python
def __init__(distribution_type: str)
```

Initialize the BrownianCoagulationStrategy.

**Parameters**

- `distribution_type` (`str`) — String specifying the distribution type (e.g., "discrete", "continuous_pdf", "particle_resolved").

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L67-L74)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.dimensionless_kernel

*method*

```python
def dimensionless_kernel(diffusive_knudsen: NDArray[np.float64], coulomb_potential_ratio: NDArray[np.float64]) -> NDArray[np.float64]
```

Not implemented for BrownianCoagulationStrategy.

This method raises NotImplementedError since dimensionless
Brownian kernels are not defined here.

**Parameters**

- `diffusive_knudsen` (`NDArray[np.float64]`) — Knudsen number array (unused).
- `coulomb_potential_ratio` (`NDArray[np.float64]`) — Coulomb ratio array (unused).

**Raises**

- `NotImplementedError` — Always, as no dimensionless kernel is
provided.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L76-L99)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.kernel

*method*

```python
def kernel(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Calculate the dimensioned Brownian coagulation kernel.

Leverages the `get_brownian_kernel_via_system_state` function to
compute the kernel, which accounts for particle size, temperature,
and pressure. The kernel typically has units of volume per time.

**Parameters**

- `particle` (`ParticleData`) — Particle data containing the distribution and density or mass data needed for the kernel calculation.
- `temperature` (`float`) — System temperature in Kelvin.
- `pressure` (`float`) — System pressure in Pascals.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Brownian coagulation kernel values. Shape depends on the underlying distribution.

**Examples**

```py
kernel_matrix = brownian_strat.kernel(particle_rep, 300, 101325)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/brownian_coagulation_strategy.py#L101-L134)

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.coulomb_potential_ratio

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

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.diffusive_knudsen

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

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.friction_factor

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

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.gain_rate

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

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.loss_rate

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

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.net_rate

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

### particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy.BrownianCoagulationStrategy.step

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
