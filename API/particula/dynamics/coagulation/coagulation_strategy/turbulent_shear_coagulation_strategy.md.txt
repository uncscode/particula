# particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy

*module*

Turbulent Shear coagulation strategies and calculations.

Provides turbulence-based coagulation kernels following Saffman & Turner
(1956). This module contains classes and functions for computing the
turbulent shear coagulation rate in aerosol systems.

**Classes**

- `- TurbulentShearCoagulationStrategy ` — Implements the abstract base
class for coagulation using a turbulent shear kernel.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L1-L1)

## particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L29-L29)

## particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy

*class*

```python
class TurbulentShearCoagulationStrategy(CoagulationStrategyABC)
```

Bases: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

Turbulent shear coagulation strategy for aerosol particles.

Implements the Saffman & Turner (1956) turbulent shear coagulation kernel,
extending the base `CoagulationStrategyABC` class to provide a physically
consistent model of coagulation in turbulent flow.

**Attributes**

- `distribution_type` — The type of particle distribution for coagulation ("discrete", "continuous_pdf", or "particle_resolved").
- `turbulent_dissipation` — Turbulent kinetic energy dissipation rate [m^2/s^3].
- `fluid_density` — Fluid density [kg/m^3].

- set_turbulent_dissipation : Set the turbulent kinetic energy dissipation
  rate.
- dimensionless_kernel : (Not implemented here) Raise NotImplementedError.
- kernel : Compute the turbulent shear coagulation kernel via
  Saffman-Turner approach.
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

**Examples**

```py title="Example usage of TurbulentShearCoagulationStrategy"
import particula as par
strategy = par.dynamics.TurbulentShearCoagulationStrategy(
    distribution_type="discrete",
    turbulent_dissipation=0.01,
    fluid_density=1.225,
)
# Use strategy.kernel(...) to get the coagulation kernel
```

> **References**
>
> - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops in
>   turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
>   https://doi.org/10.1017/S0022112056000020

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L32-L198)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.fluid_density

*attribute* · *instance attribute*

```python
fluid_density = fluid_density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L97-L97)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.turbulent_dissipation

*attribute* · *instance attribute*

```python
turbulent_dissipation = turbulent_dissipation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L96-L96)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = distribution_type
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L293-L293)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.particle_resolved_bins_number

*attribute* · *instance attribute*

```python
particle_resolved_bins_number = particle_resolved_kernel_bins_number
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L297-L299)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.particle_resolved_bins_per_decade

*attribute* · *instance attribute*

```python
particle_resolved_bins_per_decade = particle_resolved_kernel_bins_per_decade
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L300-L302)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.particle_resolved_radius

*attribute* · *instance attribute*

```python
particle_resolved_radius = particle_resolved_kernel_radius
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L296-L296)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L294-L294)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.use_direct_kernel

*attribute* · *instance attribute*

```python
use_direct_kernel = use_direct_kernel
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L303-L303)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.__init__

*method*

```python
def __init__(distribution_type: str, turbulent_dissipation: float, fluid_density: float)
```

Initialize the turbulent shear coagulation strategy.

**Parameters**

- `distribution_type` (`str`) — The distribution type ("discrete", "continuous_pdf", or "particle_resolved").
- `turbulent_dissipation` (`float`) — Turbulent kinetic energy dissipation rate [m^2/s^3].
- `fluid_density` (`float`) — The fluid density [kg/m^3].

**Returns**

- `result` (`None`) — No return value.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L77-L97)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.dimensionless_kernel

*method*

```python
def dimensionless_kernel(diffusive_knudsen: NDArray[np.float64], coulomb_potential_ratio: NDArray[np.float64]) -> NDArray[np.float64]
```

Compute a dimensionless kernel (H).

Not implemented for turbulent shear; raises NotImplementedError.

**Parameters**

- `diffusive_knudsen` (`NDArray[np.float64]`) — The diffusive Knudsen number [dimensionless].
- `coulomb_potential_ratio` (`NDArray[np.float64]`) — The Coulomb potential ratio [dimensionless].

**Returns**

- (`NDArray[np.float64]`) — NDArray[np.float64]: Not returned; raises error instead.

**Examples**

```py
# This method is not supported here
try:
    result = strategy.dimensionless_kernel(diff_kn, phi_ratio)
except NotImplementedError:
    print("Not implemented for turbulent shear strategy.")
```

> **References**
>
> - Saffman & Turner (1956) used dimensional forms; dimensionless
>   form is not covered.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L118-L153)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.kernel

*method*

```python
def kernel(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Compute the dimensioned turbulent shear coagulation kernel [m^3/s].

Uses the system state to calculate the Saffman-Turner (1956) kernel,
which depends on the dissipation rate of turbulent kinetic energy,
fluid density, and particle radius.

**Parameters**

- `particle` (`ParticleData`) — The particle data instance to retrieve particle radii.
- `temperature` (`float`) — The system temperature [K].
- `pressure` (`float`) — The system pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — float or NDArray[np.float64]: The coagulation kernel(s) [m^3/s].

**Examples**

```py title="Example usage of kernel method"
kernel_value = strategy.kernel(
    particle=particle_data,
    temperature=298.15,
    pressure=101325
)
# kernel_value could be a float or array depending on the
# native particle population
```

> **References**
>
> - Saffman, P. G., & Turner, J. S. (1956). On the collision of drops
>   in turbulent clouds. Journal of Fluid Mechanics, 1(1), 16-30.
>   https://doi.org/10.1017/S0022112056000020

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L155-L198)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.set_turbulent_dissipation

*method*

```python
def set_turbulent_dissipation(turbulent_dissipation: float)
```

Set the turbulent kinetic energy dissipation rate.

**Parameters**

- `turbulent_dissipation` (`float`) — Turbulent kinetic energy dissipation rate [m^2/s^3].

**Returns**

- `result` (`TurbulentShearCoagulationStrategy`) — This instance for method chaining.

**Examples**

```py
strategy.set_turbulent_dissipation(0.02)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_shear_coagulation_strategy.py#L99-L116)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.coulomb_potential_ratio

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.diffusive_knudsen

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.friction_factor

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.gain_rate

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.loss_rate

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.net_rate

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_shear_coagulation_strategy.TurbulentShearCoagulationStrategy.step

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
