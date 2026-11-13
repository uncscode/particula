# particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder

*module*

Turbulent DNS Coagulation Builder Module.

Provides a builder for creating `TurbulentDNSCoagulationStrategy`
instances, which handle coagulation processes under Direct Numerical
Simulation (DNS) of turbulent flows. Ensures that required parameters
such as distribution type, turbulent dissipation, fluid density,
reynolds_lambda (Taylor-scale Reynolds number), and relative_velocity
are supplied.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L1-L1)

## particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L28-L28)

## particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder

*class*

```python
class TurbulentDNSCoagulationBuilder(BuilderABC, coagulation_builder_mixin.BuilderDistributionTypeMixin, coagulation_builder_mixin.BuilderTurbulentDissipationMixin, coagulation_builder_mixin.BuilderFluidDensityMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin`

Turbulent DNS coagulation builder class.

Creates and configures a `TurbulentDNSCoagulationStrategy` to simulate
coagulation in turbulent flow fields using Direct Numerical Simulation
parameters. This builder enforces that the required parameters
(distribution_type, turbulent_dissipation, fluid_density, reynolds_lambda,
relative_velocity) are set prior to building the strategy.

**Attributes**

- `distribution_type` — The particle distribution type ("discrete", "continuous_pdf", or "particle_resolved").
- `turbulent_dissipation` — Rate of turbulent energy dissipation (m²/s³).
- `fluid_density` — Fluid density in kg/m³ (e.g., air density).
- `reynolds_lambda` — Taylor-scale Reynolds number (dimensionless).
- `relative_velocity` — Relative velocity in m/s (particle vs. airflow).

Methods:
- set_distribution_type : Set the distribution type.
- set_turbulent_dissipation : Set the turbulent dissipation rate.
- set_fluid_density : Set the fluid density.
- set_reynolds_lambda : Set the Taylor-scale Reynolds number.
- set_relative_velocity : Set the relative velocity.
- build : Validate parameters and return a
  `TurbulentDNSCoagulationStrategy`.

**Examples**

```py title="Turbulent DNS Builder Example"
builder = TurbulentDNSCoagulationBuilder()
builder.set_distribution_type("discrete")
builder.set_turbulent_dissipation(1e-3)
builder.set_fluid_density(1.225)
builder.set_reynolds_lambda(250.)
builder.set_relative_velocity(0.5, "m/s")
strategy = builder.build()
# Now 'strategy' can be used to compute DNS-based coagulation rates.

> **References**
>
> - Saffman, P. G., & Turner, J. S. (1956) "On the collision of drops
>   in turbulent clouds." Journal of Fluid Mechanics, 1(1): 16–30.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L32-L182)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.relative_velocity

*attribute* · *instance attribute*

```python
relative_velocity = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L100-L100)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.reynolds_lambda

*attribute* · *instance attribute*

```python
reynolds_lambda = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L99-L99)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L50-L50)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.turbulent_dissipation

*attribute* · *instance attribute*

```python
turbulent_dissipation = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L163-L163)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.fluid_density

*attribute* · *instance attribute*

```python
fluid_density = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L227-L227)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Turbulent DNS coagulation builder.

Sets up the builder with required parameters for creating a
TurbulentDNSCoagulationStrategy, including distribution type,
turbulent dissipation, fluid density, Reynolds lambda, and
relative velocity.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L79-L100)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.build

*method*

```python
def build() -> CoagulationStrategyABC
```

Construct a TurbulentDNSCoagulationStrategy.

Validates the required parameters, then instantiates a
`TurbulentDNSCoagulationStrategy` for DNS-based coagulation
calculations.

**Returns**

- `CoagulationStrategyABC` (`CoagulationStrategyABC`) — The configured DNS coagulation
- (`CoagulationStrategyABC`) — strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L164-L182)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.set_relative_velocity

*method*

```python
def set_relative_velocity(relative_velocity: float, relative_velocity_units: str)
```

Set the relative particle-airflow velocity for DNS coagulation.

This value is typically a background flow velocity or a
sedimentation-adjusted velocity, excluding turbulence.

**Parameters**

- `relative_velocity` (`float`) — Numeric value of velocity.
- `relative_velocity_units` (`str`) — Units of the velocity (e.g., "m/s").

**Returns**

- `result` (`TurbulentDNSCoagulationBuilder`) — This instance for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L136-L162)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.set_reynolds_lambda

*method*

```python
def set_reynolds_lambda(reynolds_lambda: float, reynolds_lambda_units: Optional[str] = None)
```

Set the Taylor-scale Reynolds number (Reλ).

Represents a measure of turbulence intensity in DNS flows.
When specifying units, only "dimensionless" is recognized here.
Any other unit triggers a warning and is treated as dimensionless.

**Parameters**

- `reynolds_lambda` (`float`) — Numeric value for Reλ.
- `reynolds_lambda_units` (`Optional[str]`) (default: `None`) — String indicating units (default "dimensionless").

**Returns**

- `result` (`TurbulentDNSCoagulationBuilder`) — This instance for method chaining.

**Examples**

```py
builder.set_reynolds_lambda(250.)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_dns_coagulation_builder.py#L102-L134)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.check_keys

*method*

```python
def check_keys(parameters: dict[str, Any]) -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if the keys are present and valid.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to check.

**Raises**

- `ValueError` — If any required key is missing or if trying to set
an invalid parameter.

**Examples**

```py
builder = Builder()
builder.check_keys({
    "parameter1": 1,
    "parameter2": 2,
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L53-L94)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.pre_build_check

*method*

```python
def pre_build_check() -> None
```

Inherited from: `particula.abc_builder.BuilderABC`

Check if all required attribute parameters are set before building.

**Raises**

- `ValueError` — If any required parameter is missing.

**Examples**

```py
builder = Builder()
builder.pre_build_check()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L132-L151)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> BuilderABC
```

Inherited from: `particula.abc_builder.BuilderABC`

Set parameters from a dictionary, handling any '_units' suffix.

**Parameters**

- `parameters` (`dict[str, Any]`) — The parameters dictionary to set.

**Returns**

- `BuilderABC` (`BuilderABC`) — This builder object with the set parameters.

**Raises**

- `ValueError` — If any required key is missing.
- `Warning` — If using default units for any parameter.

**Examples**

```py
builder = Builder().set_parameters({
    "parameter1": 1,
    "parameter2": 2,
    "parameter2_units": "K",
})
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L96-L130)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.set_distribution_type

*method*

```python
def set_distribution_type(distribution_type: str, distribution_type_units: Optional[str] = None)
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

Set the distribution type.

**Parameters**

- `distribution_type` (`str`) — The distribution type to be set. Options are "discrete", "continuous_pdf", "particle_resolved".
- `distribution_type_units` (`Optional[str]`) (default: `None`) — Not used.

**Returns**

- `result` (`BuilderDistributionTypeMixin`) — This instance for method chaining. distribution_type attribute.

**Raises**

- `ValueError` — If the distribution type is not valid.

**Examples**

```py title="Example of using set_distribution_type"
builder = SomeCoagulationBuilder()
builder.set_distribution_type("discrete")
# builder.distribution_type -> "discrete"
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L52-L98)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.set_turbulent_dissipation

*method*

```python
def set_turbulent_dissipation(turbulent_dissipation: float, turbulent_dissipation_units: str)
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin`

Set the turbulent dissipation rate.

**Parameters**

- `turbulent_dissipation` (`float`) — Turbulent dissipation rate.
- `turbulent_dissipation_units` (`str`) — Units of the turbulent dissipation rate. Default is *m^2/s^3*.

**Returns**

- `result` (`BuilderTurbulentDissipationMixin`) — This instance for method chaining. turbulent_dissipation attribute.

**Raises**

- `ValueError` — Must be non-negative value.

**Examples**

```py title="Example of using set_turbulent_dissipation"
builder = SomeCoagulationBuilder()
builder.set_turbulent_dissipation(1e-3, "m^2/s^3")
# builder.turbulent_dissipation -> 1e-3
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L165-L200)

### particula.dynamics.coagulation.coagulation_builder.turbulent_dns_coagulation_builder.TurbulentDNSCoagulationBuilder.set_fluid_density

*method*

```python
def set_fluid_density(fluid_density: Union[float, NDArray[np.float64]], fluid_density_units: str)
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin`

Set the density of the particle in kg/m^3.

**Parameters**

- `fluid_density` (`Union[float, NDArray[np.float64]]`) — Density of the fluid.
- `fluid_density_units` (`str`) — Units of the density. Default is *kg/m^3*

**Returns**

- `result` (`BuilderFluidDensityMixin`) — This instance for method chaining. fluid_density attribute.

**Raises**

- `ValueError` — Must be positive value.

**Examples**

```py title="Example of using set_fluid_density"
builder = SomeCoagulationBuilder()
builder.set_fluid_density(1.225, "kg/m^3")
# builder.fluid_density -> 1.225
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L229-L262)
