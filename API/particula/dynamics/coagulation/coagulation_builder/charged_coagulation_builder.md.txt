# particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder

*module*

Charged Coagulation Builder Module.

Provides a builder for creating `ChargedCoagulationStrategy` objects,
allowing electrostatic interactions in coagulation processes. Combines a
specified distribution type with a `ChargedKernelStrategyABC` to ensure
valid, flexible modeling of charged aerosol aggregation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L1-L1)

## particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder

*class*

```python
class ChargedCoagulationBuilder(BuilderABC, BuilderDistributionTypeMixin, BuilderUseDirectKernelMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`, `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderUseDirectKernelMixin`

Charged Coagulation builder class.

Creates a `ChargedCoagulationStrategy` based on a specified distribution
type and a `ChargedKernelStrategyABC` instance, enforcing the correct
parameters for modeling electrostatic interactions in aerosol coagulation.

**Attributes**

- `distribution_type` — Distribution representation ("discrete", "continuous_pdf", or "particle_resolved").
- `charged_kernel_strategy` — Instance of `ChargedKernelStrategyABC` for electrostatic kernel calculations.

Methods:
- set_distribution_type : Set the distribution type.
- set_charged_kernel_strategy : Set the charged kernel strategy.
- set_parameters : Configure parameters from a dictionary.
- build : Validate inputs and return a `ChargedCoagulationStrategy`.

**Examples**

```py title="Example of using ChargedCoagulationBuilder"
import particula as par
builder = par.dynamics.ChargedCoagulationBuilder()
builder.set_distribution_type("discrete")
builder.set_charged_kernel_strategy(charged_kernel_strategy)
coagulation_strategy = builder.build()
```

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). "Atmospheric Chemistry
>   and Physics." Wiley.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L27-L140)

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.charged_kernel_strategy

*attribute* · *instance attribute*

```python
charged_kernel_strategy = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L78-L78)

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.distribution_type

*attribute* · *instance attribute*

```python
distribution_type = None
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderDistributionTypeMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L50-L50)

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.use_direct_kernel

*attribute* · *instance attribute*

```python
use_direct_kernel = False
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderUseDirectKernelMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L114-L114)

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Charged coagulation builder.

Sets up the builder with required parameters for creating a
ChargedCoagulationStrategy, including distribution type and
charged kernel strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L64-L78)

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.build

*method*

```python
def build() -> CoagulationStrategyABC
```

Validate and return the ChargedCoagulationStrategy object.

This method checks whether all required parameters have been
specified (e.g., distribution type, charged kernel strategy)
before creating a `ChargedCoagulationStrategy`.

**Returns**

- `CoagulationStrategyABC` (`CoagulationStrategyABC`) — The properly configured charged coagulation strategy.

**Examples**

```py title="Example of using ChargedCoagulationBuilder build"
import particula as par
builder = ChargedCoagulationBuilder()
builder.set_distribution_type("discrete")
builder.set_charged_kernel_strategy(charged_kernel_strategy)
charged_strategy = builder.build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L114-L140)

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.set_charged_kernel_strategy

*method*

```python
def set_charged_kernel_strategy(charged_kernel_strategy: ChargedKernelStrategyABC, charged_kernel_strategy_units: Optional[str] = None)
```

Set the charged kernel strategy for electrostatic coagulation.

**Parameters**

- `charged_kernel_strategy` (`ChargedKernelStrategyABC`) — An instance of `ChargedKernelStrategyABC`.
- `charged_kernel_strategy_units` (`Optional[str]`) (default: `None`) — For interface consistency, unused.

**Raises**

- `ValueError` — If the kernel strategy is invalid or units passed
are unsupported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/charged_coagulation_builder.py#L80-L112)

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.check_keys

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

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.pre_build_check

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

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.set_parameters

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

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.set_distribution_type

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

### particula.dynamics.coagulation.coagulation_builder.charged_coagulation_builder.ChargedCoagulationBuilder.set_use_direct_kernel

*method*

```python
def set_use_direct_kernel(value: bool)
```

Inherited from: `particula.dynamics.coagulation.coagulation_builder.coagulation_builder_mixin.BuilderUseDirectKernelMixin`

Set the direct kernel evaluation flag.

**Parameters**

- `value` (`bool`) — True to compute direct pairwise kernels for particle-resolved coagulation.

**Returns**

- `result` (`BuilderUseDirectKernelMixin`) — This instance for method chaining.

**Raises**

- `ValueError` — If value is not a boolean.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/coagulation_builder/coagulation_builder_mixin.py#L116-L135)
