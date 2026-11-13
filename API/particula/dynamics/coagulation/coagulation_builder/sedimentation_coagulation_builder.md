# particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder

*module*

Sedimentation Coagulation builder class.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/sedimentation_coagulation_builder.py#L1-L1)

## particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder

*class*

```python
class SedimentationCoagulationBuilder(BuilderABC, coagulation_builder_mixin.BuilderDistributionTypeMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

Sedimentation Coagulation builder class.

Creates a `SedimentationCoagulationStrategy` given a distribution type
(e.g., "discrete", "continuous_pdf", or "particle_resolved"). Ensures
the required parameters are set before building the strategy.

**Attributes**

- `distribution_type` — Representation of the particle size distribution.

Methods:
- set_distribution_type : Set the distribution type.
- build : Validate parameters and return a
  SedimentationCoagulationStrategy instance.

**Examples**

```py title="Example of using SedimentationCoagulationBuilder"
import particula as par
builder = SedimentationCoagulationBuilder()
builder.set_distribution_type("discrete")
strategy = builder.build()
# strategy is now a SedimentationCoagulationStrategy instance
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/sedimentation_coagulation_builder.py#L16-L60)

### particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L50-L50)

### particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Sedimentation coagulation builder.

Sets up the builder with required parameters for creating a
SedimentationCoagulationStrategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/sedimentation_coagulation_builder.py#L45-L53)

### particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder.build

*method*

```python
def build() -> CoagulationStrategyABC
```

Validate and return SedimentationCoagulationStrategy object.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/sedimentation_coagulation_builder.py#L55-L60)

### particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder.check_keys

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

### particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder.pre_build_check

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

### particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder.set_parameters

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

### particula.dynamics.coagulation.coagulation_builder.sedimentation_coagulation_builder.SedimentationCoagulationBuilder.set_distribution_type

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
