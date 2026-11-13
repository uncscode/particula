# particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder

*module*

Brownian Coagulation Builder Module.

Provides a builder for creating `BrownianCoagulationStrategy` objects
based on a chosen particle distribution type (e.g., "discrete",
"continuous_pdf", or "particle_resolved"). Validates that required
parameters are set before returning the final strategy.

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). "Atmospheric Chemistry
>   and Physics." Wiley.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/brownian_coagulation_builder.py#L1-L1)

## particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder

*class*

```python
class BrownianCoagulationBuilder(BuilderABC, BuilderDistributionTypeMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

Brownian Coagulation builder class.

Creates a `BrownianCoagulationStrategy` given a distribution type
(e.g., "discrete", "continuous_pdf", or "particle_resolved"). Ensures
the required parameters are set before building the strategy.

**Attributes**

- `distribution_type` — Representation of the particle size distribution.

Methods:
- set_distribution_type : Assign the distribution type.
- set_parameters : Inherited from BuilderABC to set multiple
  parameters via a dict.
- build : Validate and return a `BrownianCoagulationStrategy`.

**Examples**

```py title="Example of using BrownianCoagulationBuilder"
import particula as par
builder = BrownianCoagulationBuilder()
builder.set_distribution_type("discrete")
strategy = builder.build()
# strategy is now a BrownianCoagulationStrategy instance
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/brownian_coagulation_builder.py#L26-L80)

### particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L50-L50)

### particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Brownian coagulation builder.

Sets up the builder with required parameters for creating a
BrownianCoagulationStrategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/brownian_coagulation_builder.py#L56-L64)

### particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder.build

*method*

```python
def build() -> CoagulationStrategyABC
```

Validate and return the BrownianCoagulationStrategy object.

Checks that all required parameters (e.g., distribution_type) are set
before creating and returning a `BrownianCoagulationStrategy`.

**Returns**

- `BrownianCoagulationStrategy` (`CoagulationStrategyABC`) — The newly created
- (`CoagulationStrategyABC`) — Brownian coagulation strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/brownian_coagulation_builder.py#L66-L80)

### particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder.check_keys

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

### particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder.pre_build_check

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

### particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder.set_parameters

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

### particula.dynamics.coagulation.coagulation_builder.brownian_coagulation_builder.BrownianCoagulationBuilder.set_distribution_type

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
