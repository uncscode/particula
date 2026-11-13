# particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin

*module*

Coagulation Builder Mixin Classes.

Provides reusable mixin classes for building coagulation strategies
with validated inputs (e.g., distribution type, turbulent dissipation,
and fluid density). These mixins can be combined to form full-featured
coagulation builders, ensuring correct parameter values are passed to
the final coagulation strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L1-L1)

## particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L21-L21)

## particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin

*class*

```python
class BuilderDistributionTypeMixin
```

Mixin class for distribution type in coagulation strategies.

Provides an interface to set the distribution type for coagulation
strategies. Ensures the chosen `distribution_type` is valid.

**Attributes**

- `distribution_type` — Stores the selected distribution type (e.g., "discrete", "continuous_pdf", "particle_resolved").

Methods:
- set_distribution_type : Set and validate the distribution type.

**Examples**

```py title="Example of using BuilderDistributionTypeMixin"
builder = SomeCoagulationBuilder()
builder.set_distribution_type("discrete")
# builder.distribution_type -> "discrete"
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L24-L98)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L50-L50)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin.__init__

*method*

```python
def __init__()
```

Initialize the distribution type mixin.

Sets the distribution_type attribute to None, to be configured later.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L45-L50)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin.set_distribution_type

*method*

```python
def set_distribution_type(distribution_type: str, distribution_type_units: Optional[str] = None)
```

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

## particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin

*class*

```python
class BuilderFluidDensityMixin
```

Mixin class for fluid density parameters.

Adds methods and attributes for setting and validating fluid
density in coagulation strategies.

**Attributes**

- `fluid_density` — Numeric value representing fluid density in kg/m^3 (default units).

Methods:
- set_fluid_density : Set and validate the fluid density.

**Examples**

```py title="Example of using BuilderFluidDensityMixin"
builder.set_fluid_density(1.225, "kg/m^3")
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L203-L262)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin.fluid_density

*attribute* · *instance attribute*

```python
fluid_density = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L227-L227)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin.__init__

*method*

```python
def __init__()
```

Initialize the fluid density mixin.

Sets the fluid_density attribute to None, to be configured later.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L222-L227)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin.set_fluid_density

*method*

```python
def set_fluid_density(fluid_density: Union[float, NDArray[np.float64]], fluid_density_units: str)
```

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

## particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin

*class*

```python
class BuilderTurbulentDissipationMixin
```

Mixin class for turbulent shear parameters.

Adds methods and attributes for setting and validating
turbulent dissipation parameters in coagulation strategies.

**Attributes**

- `turbulent_dissipation` — Numeric value of the energy dissipation rate in m^2/s^3 (default units).

- set_turbulent_dissipation : Set and validate the turbulent
    dissipation rate.

**Examples**

```py title="Example of using BuilderTurbulentDissipationMixin"
builder.set_turbulent_dissipation(1e-3, "m^2/s^3")
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L138-L200)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin.turbulent_dissipation

*attribute* · *instance attribute*

```python
turbulent_dissipation = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L163-L163)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin.__init__

*method*

```python
def __init__()
```

Initialize the turbulent dissipation mixin.

Sets the turbulent_dissipation attribute to None.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L158-L163)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin.set_turbulent_dissipation

*method*

```python
def set_turbulent_dissipation(turbulent_dissipation: float, turbulent_dissipation_units: str)
```

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

## particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderUseDirectKernelMixin

*class*

```python
class BuilderUseDirectKernelMixin
```

Mixin class for direct kernel evaluation in coagulation strategies.

**Attributes**

- `use_direct_kernel` — Whether to compute particle-resolved kernels directly instead of using interpolation.

Methods:
- set_use_direct_kernel : Set the direct kernel evaluation flag.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L101-L135)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderUseDirectKernelMixin.use_direct_kernel

*attribute* · *instance attribute*

```python
use_direct_kernel = False
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L114-L114)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderUseDirectKernelMixin.__init__

*method*

```python
def __init__()
```

Initialize the direct kernel mixin.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L112-L114)

### particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderUseDirectKernelMixin.set_use_direct_kernel

*method*

```python
def set_use_direct_kernel(value: bool)
```

Set the direct kernel evaluation flag.

**Parameters**

- `value` (`bool`) — True to compute direct pairwise kernels for particle-resolved coagulation.

**Returns**

- `result` (`BuilderUseDirectKernelMixin`) — This instance for method chaining.

**Raises**

- `ValueError` — If value is not a boolean.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L116-L135)
