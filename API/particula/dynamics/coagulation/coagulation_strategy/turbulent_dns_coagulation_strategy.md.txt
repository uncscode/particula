# particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy

*module*

Turbulent DNS coagulation strategies for particles above 1 µm.

Implements the DNS-based coagulation kernel from Ayala et al. (2008) and
adjusted for typical atmospheric or industrial conditions.
Provides classes and methods to compute collision rates under turbulent
dissipation using direct numerical simulation (DNS) approaches.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L1-L1)

## particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L27-L27)

## particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy

*class*

```python
class TurbulentDNSCoagulationStrategy(CoagulationStrategyABC)
```

Bases: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

Turbulent DNS coagulation strategy for aerosols.

Implements methods from `CoagulationStrategyABC`, applying the
turbulent DNS kernel following Ayala et al. (2008). Suitable for
coagulation of particles larger than 1 µm in turbulent flow fields.

**Attributes**

- `distribution_type` — The particle distribution type ("discrete", "continuous_pdf", or "particle_resolved").
- `turbulent_dissipation` — Turbulent kinetic energy dissipation [m^2/s^3] used in DNS fits (examples: 0.001, 0.01, 0.04).
- `fluid_density` — The fluid (air) density [kg/m^3].
- `reynolds_lambda` — Reynolds lambda of air (e.g., 23 or 74).
- `relative_velocity` — Relative velocity of the air [m/s] for collisions.

Methods:
- set_turbulent_dissipation : Change turbulent dissipation rate.
- set_reynolds_lambda : Update the Reynolds lambda.
- set_relative_velocity : Update the relative velocity.
- dimensionless_kernel : Raise NotImplementedError for DNS approach.
- kernel : Return the DNS-based coagulation kernel.
- loss_rate : Calculate the coagulation loss rate.
- gain_rate : Calculate the coagulation gain rate.
- net_rate : Get the net coagulation rate (gain - loss).
- step : Perform a single step of coagulation.
- diffusive_knudsen : Calculate the diffusive Knudsen number.
- coulomb_potential_ratio : Compute Coulomb potential ratio.
- friction_factor : Compute the effective friction factor.

**Examples**

```py title="Example usage of TurbulentDNSCoagulationStrategy"
import particula as par
strategy = par.dynamics.TurbulentDNSCoagulationStrategy(
    distribution_type="discrete",
    turbulent_dissipation=0.01,
    fluid_density=1.225,
    reynolds_lambda=23,
    relative_velocity=0.5
)
# Use strategy.kernel(...) to compute the DNS-based kernel
```

References:
- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the
  geometric collision rate of sedimenting droplets. Part 2. Theory and
  parameterization. New Journal of Physics, 10.
  [DOI](https://doi.org/10.1088/1367-2630/10/7/075016)

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L30-L244)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.fluid_density

*attribute* · *instance attribute*

```python
fluid_density = fluid_density
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L107-L107)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.relative_velocity

*attribute* · *instance attribute*

```python
relative_velocity = relative_velocity
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L109-L109)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.reynolds_lambda

*attribute* · *instance attribute*

```python
reynolds_lambda = reynolds_lambda
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L108-L108)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.turbulent_dissipation

*attribute* · *instance attribute*

```python
turbulent_dissipation = turbulent_dissipation
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L106-L106)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = distribution_type
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L293-L293)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.particle_resolved_bins_number

*attribute* · *instance attribute*

```python
particle_resolved_bins_number = particle_resolved_kernel_bins_number
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L297-L299)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.particle_resolved_bins_per_decade

*attribute* · *instance attribute*

```python
particle_resolved_bins_per_decade = particle_resolved_kernel_bins_per_decade
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L300-L302)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.particle_resolved_radius

*attribute* · *instance attribute*

```python
particle_resolved_radius = particle_resolved_kernel_radius
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L296-L296)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L294-L294)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.use_direct_kernel

*attribute* · *instance attribute*

```python
use_direct_kernel = use_direct_kernel
```

Inherited from: `particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc.CoagulationStrategyABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/coagulation_strategy_abc.py#L303-L303)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.__init__

*method*

```python
def __init__(distribution_type: str, turbulent_dissipation: float, fluid_density: float, reynolds_lambda: float, relative_velocity: float)
```

Initialize the TurbulentDNSCoagulationStrategy.

**Parameters**

- `distribution_type` (`str`) — The distribution type ("discrete", "continuous_pdf", or "particle_resolved").
- `turbulent_dissipation` (`float`) — Turbulent kinetic energy dissipation [m^2/s^3].
- `fluid_density` (`float`) — The fluid density [kg/m^3].
- `reynolds_lambda` (`float`) — Reynolds lambda or characteristic Reynolds number.
- `relative_velocity` (`float`) — Relative velocity of the flow [m/s].

**Returns**

- `result` (`None`) — No return value.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L81-L109)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.dimensionless_kernel

*method*

```python
def dimensionless_kernel(diffusive_knudsen: NDArray[np.float64], coulomb_potential_ratio: NDArray[np.float64]) -> NDArray[np.float64]
```

Compute or return the dimensionless kernel (H).

Not implemented for DNS-based approaches, so raises
NotImplementedError.

**Parameters**

- `diffusive_knudsen` (`NDArray[np.float64]`) — The diffusive Knudsen number(s) [dimensionless].
- `coulomb_potential_ratio` (`NDArray[np.float64]`) — The Coulomb potential ratio(s) [dimensionless].

**Returns**

- `None` (`NDArray[np.float64]`) — Raises NotImplementedError instead.

**Raises**

- `NotImplementedError` — This strategy does not support
dimensionless kernels.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L165-L193)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.kernel

*method*

```python
def kernel(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Compute the DNS-based coagulation kernel [m^3/s].

Uses the `get_turbulent_dns_kernel_ao2008_via_system_state` function to
calculate collision rates following Ayala et al. (2008). This approach
accounts for turbulent dissipation, fluid density, Reynolds lambda,
and relative velocity.

**Parameters**

- `particle` (`ParticleData`) — The particle data whose radii and density are needed.
- `temperature` (`float`) — The temperature of the system [K].
- `pressure` (`float`) — The system pressure [Pa] (unused here, but included for interface consistency).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The DNS-based coagulation kernel(s).

**Examples**

```py
kernel_values = strategy.kernel(
    particle=my_particle,
    temperature=298.15,
    pressure=101325
)
# kernel_values may be a float or array, depending on the
# distribution
```

References:
- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence
  on the geometric collision rate of sedimenting droplets. Part 2.
  New Journal of Physics, 10.
  [DOI](https://doi.org/10.1088/1367-2630/10/7/075016)

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L195-L244)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.set_relative_velocity

*method*

```python
def set_relative_velocity(relative_velocity: float)
```

Set the relative velocity of the flow [m/s].

**Parameters**

- `relative_velocity` (`float`) — Relative velocity in [m/s].

**Returns**

- `result` (`TurbulentDNSCoagulationStrategy`) — This instance for method chaining.

**Examples**

```py
strategy.set_relative_velocity(0.8)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L147-L163)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.set_reynolds_lambda

*method*

```python
def set_reynolds_lambda(reynolds_lambda: float)
```

Set the Reynolds lambda value.

**Parameters**

- `reynolds_lambda` (`float`) — Reynolds lambda [dimensionless].

**Returns**

- `result` (`TurbulentDNSCoagulationStrategy`) — This instance for method chaining.

**Examples**

```py
strategy.set_reynolds_lambda(74)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L129-L145)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.set_turbulent_dissipation

*method*

```python
def set_turbulent_dissipation(turbulent_dissipation: float)
```

Set the turbulent kinetic energy dissipation rate.

**Parameters**

- `turbulent_dissipation` (`float`) — Turbulent dissipation [m^2/s^3].

**Returns**

- `result` (`TurbulentDNSCoagulationStrategy`) — This instance for method chaining.

**Examples**

```py
strategy.set_turbulent_dissipation(0.02)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_strategy/turbulent_dns_coagulation_strategy.py#L111-L127)

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.coulomb_potential_ratio

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.diffusive_knudsen

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.friction_factor

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.gain_rate

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.loss_rate

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.net_rate

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

### particula.dynamics.coagulation.coagulation_strategy.turbulent_dns_coagulation_strategy.TurbulentDNSCoagulationStrategy.step

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
