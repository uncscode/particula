# particula.particles.activity_builders

*module*

Builder class for Activity objects with validation and error handling.

Change to MixinMolar classes, after PR integration.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L1-L1)

## particula.particles.activity_builders.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L35-L35)

## particula.particles.activity_builders.ActivityIdealMassBuilder

*class*

```python
class ActivityIdealMassBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builds an ActivityIdealMass object for calculating activity based on
ideal mass fractions.

A concise builder for ActivityIdealMass. This class requires no extra
parameters beyond the defaults. It ensures the returned strategy follows
Raoult's Law for mass-based activities.

**Functions**

- `- build` — Validates any required parameters and returns the strategy.

**Examples**

```py title="Example Usage"
import particula as par
builder = par.particles.ActivityIdealMassBuilder()
strategy = builder.build()
result = strategy.activity([1.0, 2.0, 3.0])
# result -> ...
```

References:
- "Raoult's Law,"
    [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L38-L86)

### particula.particles.activity_builders.ActivityIdealMassBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.activity_builders.ActivityIdealMassBuilder.__init__

*method*

```python
def __init__()
```

Initialize the ActivityIdealMass builder.

Sets up the builder with no required parameters for creating
an ActivityIdealMass strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L63-L70)

### particula.particles.activity_builders.ActivityIdealMassBuilder.build

*method*

```python
def build() -> ActivityStrategy
```

Validate and return an ActivityIdealMass strategy instance.

**Returns**

- `ActivityIdealMass` (`ActivityStrategy`) — The validated strategy for ideal mass-based activity calculations.

**Examples**

```py title="Build Method Example"
builder = par.particles.ActivityIdealMassBuilder()
mass_activity_strategy = builder.build()
# Use mass_activity_strategy.activity(...)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L72-L86)

### particula.particles.activity_builders.ActivityIdealMassBuilder.check_keys

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

### particula.particles.activity_builders.ActivityIdealMassBuilder.pre_build_check

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

### particula.particles.activity_builders.ActivityIdealMassBuilder.set_parameters

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

## particula.particles.activity_builders.ActivityIdealMolarBuilder

*class*

```python
class ActivityIdealMolarBuilder(BuilderABC, BuilderMolarMassMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`

Builds an ActivityIdealMolar object for calculating activity from
ideal mole fractions.

This builder sets up any required parameters (e.g., molar mass) and
creates an ActivityIdealMolar strategy. Uses Raoult's Law in terms
of mole fraction.

**Attributes**

- `molar_mass` (`Optional[Union[float, NDArray[np.float64]]]`) — Molar mass for each species, in kilograms per mole.

**Functions**

- `- set_molar_mass` — Assigns the molar masses (with unit
handling).
- `- set_parameters` — Batch-assign parameters from a dictionary.
- `- build` — Finalizes the builder and returns the strategy.

**Examples**

```py title="Example Usage"
import particula as par
builder = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(0.01815, "kg/mol")
)
strategy = builder.build()
result = strategy.activity([1.0, 2.0, 3.0])
# result -> ...
```

> **References**
>
> - "Raoult's Law,"
> [Wikipedia](https://en.wikipedia.org/wiki/Raoult%27s_law).

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L89-L159)

### particula.particles.activity_builders.ActivityIdealMolarBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.activity_builders.ActivityIdealMolarBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.particles.activity_builders.ActivityIdealMolarBuilder.__init__

*method*

```python
def __init__()
```

Initialize the ActivityIdealMolar builder.

Sets up the builder with required molar_mass parameter for
creating an ActivityIdealMolar strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L123-L131)

### particula.particles.activity_builders.ActivityIdealMolarBuilder.build

*method*

```python
def build() -> ActivityStrategy
```

Validate parameters and create an ActivityIdealMolar strategy.

Ensures molar_mass is properly configured before building.

**Returns**

- `ActivityIdealMolar` (`ActivityStrategy`) — An ideal strategy based on mole fractions.

**Examples**

```py title="Build Method Example"
builder = (
    par.particles.ActivityIdealMolarBuilder()
    .set_molar_mass(0.028, "kg/mol")
)
molar_activity_strategy = builder.build()
# molar_activity_strategy.activity(...)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L133-L159)

### particula.particles.activity_builders.ActivityIdealMolarBuilder.check_keys

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

### particula.particles.activity_builders.ActivityIdealMolarBuilder.pre_build_check

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

### particula.particles.activity_builders.ActivityIdealMolarBuilder.set_parameters

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

### particula.particles.activity_builders.ActivityIdealMolarBuilder.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[float, NDArray[np.float64]], molar_mass_units: str)
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

Set the molar mass in kilograms per mole.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Positive molar mass value to store.
- `molar_mass_units` (`str`) — Units for ``molar_mass``. Defaults to ``"kg/mol"``.

**Returns**

- `self` (`BuilderMolarMassMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L151-L173)

## particula.particles.activity_builders.ActivityKappaParameterBuilder

*class*

```python
class ActivityKappaParameterBuilder(BuilderABC, BuilderDensityMixin, BuilderMolarMassMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderDensityMixin`, `particula.builder_mixin.BuilderMolarMassMixin`

Builds an ActivityKappaParameter object for non-ideal activity
calculations.

This builder requires kappa, density, molar_mass, and water_index.
Kappa is the hygroscopicity parameter, used to capture non-ideal
behavior. The optional water_index identifies which species is water.

**Attributes**

- `kappa` — NDArray of kappa parameters for each species.
- `density` (`Optional[Union[float, NDArray[np.float64]]]`) — NDArray of densities, in kilograms per cubic meter.
- `molar_mass` (`Optional[Union[float, NDArray[np.float64]]]`) — NDArray of molar masses, in kilograms per mole.
- `water_index` — Integer index of the water species.

**Functions**

- `- set_kappa` — Assigns kappa values (must be nonnegative).
- `- set_water_index` — Sets the index of the water species.
- `- set_density` — Assigns density values (with unit handling).
- `- set_molar_mass` — Assigns molar mass values (with unit
handling).
- `- set_parameters` — Batch-assign parameters from a dictionary.
- `- build` — Finalizes checks and returns the strategy.

**Examples**

```py title="Example Usage"
import particula as par
import numpy as np

builder = (
    par.particles.ActivityKappaParameterBuilder()
    .set_kappa(np.array([0.1, 0.0]))
    .set_density(np.array([1000, 1200]), "kg/m^3"))
    .set_molar_mass(np.array([0.018, 0.058]), "kg/mol")
    .set_water_index(0)
)
strategy = builder.build()
result = strategy.activity(np.array([1.0, 2.0]))
# result -> ...
```

> **References**
>
> - Petters, M. D., and Kreidenweis, S. M. (2007).
>   "A single parameter representation of hygroscopic growth and
>    cloud condensation nucleus activity," Atmospheric Chemistry
>    and Physics, 7(8), 1961–1971.
>    [DOI](https://doi.org/10.5194/acp-7-1961-2007)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L162-L290)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.kappa

*attribute* · *instance attribute*

```python
kappa = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L223-L223)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.water_index

*attribute* · *instance attribute*

```python
water_index = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L224-L224)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.density

*attribute* · *instance attribute*

```python
density: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L34-L34)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L149-L149)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.__init__

*method*

```python
def __init__()
```

Initialize the ActivityKappaParameter builder.

Sets up the builder with required parameters for creating an
ActivityKappaParameter strategy, including kappa, density,
molar mass, and water index.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L212-L224)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.build

*method*

```python
def build() -> ActivityStrategy
```

Validate parameters and build ActivityKappaParameter strategy.

**Returns**

- `ActivityKappaParameter` (`ActivityStrategy`) — The non-ideal activity strategy utilizing the kappa hygroscopic parameter.

**Examples**

```py title="Build Method Example"
kappa_activity_strategy = (
    par.particles.ActivityKappaParameterBuilder()
    .set_kappa([0.1, 0.2])
    .set_density([1000, 1200], "kg/m^3")
    .set_molar_mass([0.018, 0.046], "kg/mol")
    .set_water_index(0)
    .build()
)
# kappa_activity_strategy ...
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L264-L290)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.set_kappa

*method*

```python
def set_kappa(kappa: Union[float, NDArray[np.float64]], kappa_units: Optional[str] = None)
```

Set the kappa parameter for the activity calculation.

**Parameters**

- `kappa` (`Union[float, NDArray[np.float64]]`) — The kappa parameter for the activity calculation.
- `kappa_units` (`Optional[str]`) (default: `None`) — Not used. (for interface consistency)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L226-L244)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.set_water_index

*method*

```python
def set_water_index(water_index: int, water_index_units: Optional[str] = None)
```

Set the array index of the species.

**Parameters**

- `water_index` (`int`) — The array index of the species.
- `water_index_units` (`Optional[str]`) (default: `None`) — Not used. (for interface consistency)

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L246-L262)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.check_keys

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

### particula.particles.activity_builders.ActivityKappaParameterBuilder.pre_build_check

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

### particula.particles.activity_builders.ActivityKappaParameterBuilder.set_parameters

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

### particula.particles.activity_builders.ActivityKappaParameterBuilder.set_density

*method*

```python
def set_density(density: Union[float, NDArray[np.float64]], density_units: str)
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

Set the particle density in kilograms per cubic meter.

**Parameters**

- `density` (`Union[float, NDArray[np.float64]]`) — Positive density value to store.
- `density_units` (`str`) — Units for ``density``. Defaults to ``"kg/m^3"`` and accepts any unit supported by the converter.

**Returns**

- `self` (`BuilderDensityMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L36-L56)

### particula.particles.activity_builders.ActivityKappaParameterBuilder.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[float, NDArray[np.float64]], molar_mass_units: str)
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

Set the molar mass in kilograms per mole.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Positive molar mass value to store.
- `molar_mass_units` (`str`) — Units for ``molar_mass``. Defaults to ``"kg/mol"``.

**Returns**

- `self` (`BuilderMolarMassMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L151-L173)

## particula.particles.activity_builders.ActivityNonIdealBinaryBuilder

*class*

```python
class ActivityNonIdealBinaryBuilder(BuilderABC, BuilderMolarMassMixin, BuilderDensityMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.builder_mixin.BuilderDensityMixin`

Builder for ActivityNonIdealBinary strategy using BAT model.

Provides a fluent interface to configure the binary non-ideal
activity strategy with validation of required parameters and optional
functional group metadata.

Required parameters (via setters or ``set_parameters``):
    - molar_mass: Organic molar mass in kg/mol.
    - oxygen2carbon: Oxygen to carbon atomic ratio (dimensionless).
    - density: Organic density in kg/m^3.

> **Optional parameters**
>
> - functional_group: Functional group identifier string or list.

**Examples**

```pycon
>>> import particula as par
>>> builder = (
...     par.particles.ActivityNonIdealBinaryBuilder()
...     .set_molar_mass(0.200, "kg/mol")
...     .set_oxygen2carbon(0.4)
...     .set_density(1400.0, "kg/m^3")
...     .set_functional_group("carboxylic_acid")
... )
>>> strategy = builder.build()
>>> strategy.get_name()
'ActivityNonIdealBinary'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L293-L489)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.density

*attribute* · *instance attribute*

```python
density: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L331-L331)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.functional_group

*attribute* · *instance attribute*

```python
functional_group: Optional[Union[str, List[str]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L333-L333)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: Optional[Union[float, NDArray[np.float64]]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L330-L330)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.oxygen2carbon

*attribute* · *instance attribute*

```python
oxygen2carbon: Optional[float] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L332-L332)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the builder with required parameters.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L324-L333)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder._to_scalar

*method* · *staticmethod*

```python
def _to_scalar(value: Union[float, NDArray[np.float64]]) -> float
```

Convert scalar-like input to float, rejecting multi-value arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L335-L343)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.build

*method*

```python
def build() -> ActivityNonIdealBinary
```

Validate required inputs then build the strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L460-L489)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.set_functional_group

*method*

```python
def set_functional_group(functional_group: Optional[Union[str, List[str]]], functional_group_units: Optional[str] = None) -> Self
```

Set optional functional group identifier(s).

**Parameters**

- `functional_group` (`Optional[Union[str, List[str]]]`) — Functional group value; accepts None, str, or list.
- `functional_group_units` (`Optional[str]`) (default: `None`) — Ignored; warns if provided.

**Returns**

- (`Self`) — Self for fluent chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L376-L399)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.set_oxygen2carbon

*method*

```python
def set_oxygen2carbon(oxygen2carbon: Union[float, NDArray[np.float64]], oxygen2carbon_units: Optional[str] = None) -> Self
```

Set oxygen-to-carbon ratio ensuring nonnegativity.

**Parameters**

- `oxygen2carbon` (`Union[float, NDArray[np.float64]]`) — Oxygen to carbon atomic ratio (>=0).
- `oxygen2carbon_units` (`Optional[str]`) (default: `None`) — Ignored (dimensionless); warns if provided.

**Returns**

- (`Self`) — Self for fluent chaining.

**Raises**

- `ValueError` — If ``oxygen2carbon`` contains negative values.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L345-L374)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.set_parameters

*method*

```python
def set_parameters(parameters: dict[str, Any]) -> ActivityNonIdealBinaryBuilder
```

Batch assign parameters with optional units and validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/activity_builders.py#L401-L458)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.check_keys

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

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.pre_build_check

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

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[float, NDArray[np.float64]], molar_mass_units: str)
```

Inherited from: `particula.builder_mixin.BuilderMolarMassMixin`

Set the molar mass in kilograms per mole.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Positive molar mass value to store.
- `molar_mass_units` (`str`) — Units for ``molar_mass``. Defaults to ``"kg/mol"``.

**Returns**

- `self` (`BuilderMolarMassMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L151-L173)

### particula.particles.activity_builders.ActivityNonIdealBinaryBuilder.set_density

*method*

```python
def set_density(density: Union[float, NDArray[np.float64]], density_units: str)
```

Inherited from: `particula.builder_mixin.BuilderDensityMixin`

Set the particle density in kilograms per cubic meter.

**Parameters**

- `density` (`Union[float, NDArray[np.float64]]`) — Positive density value to store.
- `density_units` (`str`) — Units for ``density``. Defaults to ``"kg/m^3"`` and accepts any unit supported by the converter.

**Returns**

- `self` (`BuilderDensityMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L36-L56)
