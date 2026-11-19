# particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder

*module*

Builder for a compact-sphere dispersion and polarization strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/van_der_waals_coagulation_builder.py#L1-L1)

## particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder

*class*

```python
class VanDerWaalsCoagulationBuilder(BuilderABC, BuilderDistributionTypeMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

Build a thermal encounter strategy for a specified material.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/van_der_waals_coagulation_builder.py#L12-L52)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.hamaker_constant

*attribute* · *instance attribute*

```python
hamaker_constant = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/van_der_waals_coagulation_builder.py#L22-L22)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.relative_permittivity

*attribute* · *instance attribute*

```python
relative_permittivity = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/van_der_waals_coagulation_builder.py#L23-L23)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L50-L50)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.__init__

*method*

```python
def __init__()
```

Require both material constants and a population representation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/van_der_waals_coagulation_builder.py#L15-L23)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.build

*method*

```python
def build()
```

Return the configured coagulation strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/van_der_waals_coagulation_builder.py#L45-L52)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.set_hamaker_constant

*method*

```python
def set_hamaker_constant(hamaker_constant, hamaker_constant_units='J')
```

Set the effective Hamaker constant in joules.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/van_der_waals_coagulation_builder.py#L25-L32)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.set_relative_permittivity

*method*

```python
def set_relative_permittivity(relative_permittivity, relative_permittivity_units='dimensionless')
```

Set the dimensionless static relative permittivity.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/van_der_waals_coagulation_builder.py#L34-L43)

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.check_keys

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

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.pre_build_check

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

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.set_parameters

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

### particula.dynamics.coagulation.coagulation_builder.van_der_waals_coagulation_builder.VanDerWaalsCoagulationBuilder.set_distribution_type

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
