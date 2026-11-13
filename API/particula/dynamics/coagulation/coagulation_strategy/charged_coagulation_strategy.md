# particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy

*module*

Charged particle coagulation strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L1-L1)

## particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L22-L22)

## particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy

*class*

```python
class ChargedCoagulationStrategy(CoagulationStrategyABC)
```

Bases: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

Charged Brownian coagulation strategy using a dimensionless kernel.

This class implements the methods defined in the CoagulationStrategyABC
abstract class. A ChargedKernelStrategyABC instance is passed to define
how the dimensionless kernel is calculated. This approach allows flexible
handling of Coulomb interactions under various regimes.

**Attributes**

- `kernel_strategy` — Instance of ChargedKernelStrategyABC used to calculate dimensionless and dimensioned kernels.

- dimensionless_kernel : Compute dimensionless kernel values for
  charged coagulation.
- kernel : Convert dimensionless kernel values into a dimensioned
  coagulation kernel.
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

**Examples**

```py title="Example Usage"
import numpy as np
import particula as par
kernel_strategy = par.dynamics.HardSphereKernelStrategy()
charged_coag = par.dynamics.ChargedCoagulationStrategy(
    distribution_type="discrete", kernel_strategy=kernel_strategy
)
particle_data = par.particles.create_lognormal_binned_particle_data(
    distribution_type="discrete"
)
kernel_values = charged_coag.kernel(particle_data, 298.15, 101325)
```

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. "Atmospheric Chemistry and
>   Physics: From Air Pollution to Climate Change." Wiley, 2016.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L25-L192)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.kernel_strategy

*attribute* · *instance attribute*

```python
kernel_strategy = kernel_strategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L94-L94)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = distribution_type
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L293-L293)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.particle_resolved_bins_number

*attribute* · *instance attribute*

```python
particle_resolved_bins_number = particle_resolved_kernel_bins_number
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L297-L299)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.particle_resolved_bins_per_decade

*attribute* · *instance attribute*

```python
particle_resolved_bins_per_decade = particle_resolved_kernel_bins_per_decade
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L300-L302)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.particle_resolved_radius

*attribute* · *instance attribute*

```python
particle_resolved_radius = particle_resolved_kernel_radius
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L296-L296)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L294-L294)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.use_direct_kernel

*attribute* · *instance attribute*

```python
use_direct_kernel = use_direct_kernel
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L303-L303)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.__init__

*method*

```python
def __init__(distribution_type: str, kernel_strategy: ChargedKernelStrategyABC, use_direct_kernel: bool = False)
```

Initialize the ChargedCoagulationStrategy.

**Parameters**

- `distribution_type` (`str`) — The distribution type representing how particles are tracked (e.g., "discrete", "continuous_pdf", or "particle_resolved").
- `kernel_strategy` (`ChargedKernelStrategyABC`) — A ChargedKernelStrategyABC instance used to calculate dimensionless/dimensioned kernels for charged coagulation.
- `use_direct_kernel` (`bool`) (default: `False`) — Whether to compute kernel values directly for particle-resolved coagulation instead of using interpolation.

**Returns**

- `result` (`None`) — No return value.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L69-L94)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.dimensionless_kernel

*method*

```python
def dimensionless_kernel(diffusive_knudsen: NDArray[np.float64], coulomb_potential_ratio: NDArray[np.float64]) -> NDArray[np.float64]
```

Compute the dimensionless kernel for charged coagulation.

This method delegates computation to the provided kernel strategy. It
returns the dimensionless kernel (H) as a function of the diffusive
Knudsen number and the Coulomb potential ratio.

**Parameters**

- `diffusive_knudsen` (`NDArray[np.float64]`) — Dimensionless Knudsen number(s) describing particle diffusive behavior.
- `coulomb_potential_ratio` (`NDArray[np.float64]`) — Dimensionless ratio(s) incorporating electrostatic interactions.

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Array of dimensionless kernel values.

**Examples**

```py title="Dimensionless Kernel Example"
kn = np.array([0.1, 0.2])
phi = np.array([1.0, 2.0])
dim_kernel = charged_coag.dimensionless_kernel(kn, phi)
# dim_kernel -> array of dimensionless kernel values
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L96-L127)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.kernel

*method*

```python
def kernel(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Compute the dimensioned coagulation kernel for charged particles.

This method converts the dimensionless kernel into a dimensioned
coagulation kernel by combining Coulomb parameters, the pairwise
radii of particles, reduced mass, and friction factors.

**Parameters**

- `particle` (`ParticleData`) — Particle data containing distribution, density, and concentration data.
- `temperature` (`float`) — Float specifying the system temperature (K).
- `pressure` (`float`) — Float specifying the system pressure (Pa).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — float or NDArray[np.float64]: The dimensioned coagulation kernel value(s).

**Examples**

```py title="Dimensioned Kernel Example"
kernel_matrix = charged_coag.kernel(
    particle=my_particle, temperature=300, pressure=101325
)
# kernel_matrix -> 2D array of size (n_particles, n_particles)
```

References:
- Gopalakrishnan, R. & Hogan, C. J. "Determination of the Transition
  Regime Collision Kernel from Mean First Passage Times." Aerosol
  Science and Technology, 46: 887-899, 2012.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/charged_coagulation_strategy.py#L129-L192)

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.coulomb_potential_ratio

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

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.diffusive_knudsen

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

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.friction_factor

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

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.gain_rate

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

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.loss_rate

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

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.net_rate

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

### particula.dynamics.coagulation.coagulation_strategy.charged_coagulation_strategy.ChargedCoagulationStrategy.step

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
