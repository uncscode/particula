# particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder

*module*

Turbulent Shear Coagulation Builder Module.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_shear_coagulation_builder.py#L1-L1)

## particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder

*class*

```python
class TurbulentShearCoagulationBuilder(BuilderABC, coagulation_builder_mixin.BuilderDistributionTypeMixin, coagulation_builder_mixin.BuilderTurbulentDissipationMixin, coagulation_builder_mixin.BuilderFluidDensityMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin`

Turbulent shear coagulation builder.

Creates a TurbulentShearCoagulationStrategy that calculates coagulation
rates under turbulent flow conditions. Ensures the correct distribution
type, turbulent dissipation, and fluid density values are provided.

**Attributes**

- `distribution_type` — Type of the particle distribution ("discrete", "continuous_pdf", or "particle_resolved").
- `turbulent_dissipation` — Turbulent energy dissipation rate (m²/s³).
- `fluid_density` — Fluid density (kg/m³) for the coagulation medium.

**Functions**

- `- set_distribution_type ` — Set the distribution type.
- `- set_turbulent_dissipation ` — Set turbulent dissipation rate.
- `- set_fluid_density ` — Set fluid density.
- `- build ` — Validate parameters and return a
TurbulentShearCoagulationStrategy.

**Examples**

```py title="Turbulent Shear Coagulation Builder Example"
import particula as par
builder = par.dynamics.TurbulentShearCoagulationBuilder()
builder.set_distribution_type("discrete")
builder.set_turbulent_dissipation(1e-3)
builder.set_fluid_density(1000.)
strategy = builder.build()
# Now 'strategy' can be used to compute turbulent shear coagulation
# rates.
```

> **References**
>
> - Saffman, P. G., & Turner, J. S. (1956). "On the collision of drops
>   in turbulent clouds." J. Fluid Mech., 1, 16-30.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_shear_coagulation_builder.py#L17-L95)

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L50-L50)

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.turbulent_dissipation

*attribute* · *instance attribute*

```python
turbulent_dissipation = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderTurbulentDissipationMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L163-L163)

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.fluid_density

*attribute* · *instance attribute*

```python
fluid_density = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderFluidDensityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L227-L227)

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.__init__

*method*

```python
def __init__()
```

Initialize the TurbulentShearCoagulationBuilder.

**Returns**

- `result` (`None`) — No return value.

> **Note**
>
> Some default values may be set by the mixins to guide the user
> toward valid operation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_shear_coagulation_builder.py#L59-L78)

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.build

*method*

```python
def build() -> CoagulationStrategyABC
```

Construct a TurbulentShearCoagulationStrategy.

This method performs a final check to ensure all required parameters
have been set. It then creates and returns an instance of
TurbulentShearCoagulationStrategy.

**Returns**

- (`CoagulationStrategyABC`) — The resulting turbulent shear coagulation strategy object.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/turbulent_shear_coagulation_builder.py#L80-L95)

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.check_keys

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

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.pre_build_check

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

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.set_parameters

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

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.set_distribution_type

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

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.set_turbulent_dissipation

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

### particula.dynamics.coagulation.coagulation_builder.turbulent_shear_coagulation_builder.TurbulentShearCoagulationBuilder.set_fluid_density

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
