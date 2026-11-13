# particula.dynamics.nucleation.nucleation_builders

*module*

Build immutable, normalized public construction nucleation configurations.

The builders accept only the documented physical units and make configuration
from mappings atomic. They construct potential-rate metadata only; demand
planning/source transaction
source-demand finalization and particle or gas mutation remain in the concrete
``particle_source`` module.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L1-L1)

## particula.dynamics.nucleation.nucleation_builders._REQUIRED

*attribute* · *module attribute*

```python
_REQUIRED = ('coefficient', 'coefficient_provenance', 'precursor_number_concentration_lower', 'precursor_number_concentration_upper', 'temperature_lower', 'temperature_upper', 'injection_composition', 'formation_diameter')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L32-L41)

## particula.dynamics.nucleation.nucleation_builders._SCALARS

*attribute* · *module attribute*

```python
_SCALARS = {'coefficient', 'precursor_number_concentration_lower', 'precursor_number_concentration_upper', 'temperature_lower', 'temperature_upper', 'formation_diameter', 'saturation_lower', 'saturation_upper', 'survival_factor'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L42-L52)

## particula.dynamics.nucleation.nucleation_builders._STAGED_ATTRIBUTES

*attribute* · *module attribute*

```python
_STAGED_ATTRIBUTES = (*_SCALARS, 'coefficient_provenance', 'injection_composition', 'diameter_convention')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L53-L58)

## particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder

*class*

```python
class ActivationNucleationBuilder(_NucleationStrategyBuilder)
```

Bases: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Build immutable activation potential-rate strategies.

The activation coefficient must use ``"s^-1"`` and all retained physical
values are normalized to SI units before construction.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L470-L498)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.coefficient_units

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_units = {'s^-1'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L477-L477)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.coefficient

*attribute* · *instance attribute*

```python
coefficient: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L74-L74)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.coefficient_provenance

*attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L75-L75)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.diameter_convention

*attribute* · *instance attribute*

```python
diameter_convention: str | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L85-L85)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.formation_diameter

*attribute* · *instance attribute*

```python
formation_diameter: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L81-L81)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L80-L80)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.precursor_number_concentration_lower

*attribute* · *instance attribute*

```python
precursor_number_concentration_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L76-L76)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.precursor_number_concentration_upper

*attribute* · *instance attribute*

```python
precursor_number_concentration_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L77-L77)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.saturation_lower

*attribute* · *instance attribute*

```python
saturation_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L82-L82)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.saturation_upper

*attribute* · *instance attribute*

```python
saturation_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L83-L83)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.survival_factor

*attribute* · *instance attribute*

```python
survival_factor: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L84-L84)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.temperature_lower

*attribute* · *instance attribute*

```python
temperature_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L78-L78)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.temperature_upper

*attribute* · *instance attribute*

```python
temperature_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L79-L79)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.build

*method*

```python
def build() -> ActivationNucleationStrategy
```

Build a fully validated immutable activation strategy.

**Returns**

- (`ActivationNucleationStrategy`) — Activation strategy with normalized metadata and a survival default
- (`ActivationNucleationStrategy`) — of ``1.0`` when no survival factor was supplied.

**Raises**

- `ValueError` — If required builder fields or paired saturation bounds
are missing or invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L479-L498)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.__init__

*method*

```python
def __init__() -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Initialize an unconfigured strategy builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L71-L85)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder._configuration

*method*

```python
def _configuration() -> tuple[NucleationValidityDomain, InjectionComposition, FormationMetadata, float]
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L416-L467)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder._converted

*method* · *staticmethod*

```python
def _converted(value: object, unit: object, accepted: set[str], name: str) -> float
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L93-L106)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder._set_mapping_scalar

*method*

```python
def _set_mapping_scalar(key: str, value: object, units: object) -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L353-L374)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder._set_positive

*method*

```python
def _set_positive(name: str, value: object, units: object, accepted: set[str]) -> float
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L184-L199)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder._unit

*method* · *staticmethod*

```python
def _unit(unit: object, accepted: set[str], name: str) -> str
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L87-L91)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder._validate_schema

*method* · *staticmethod*

```python
def _validate_schema(parameters: dict[str, object]) -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L328-L351)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_coefficient

*method*

```python
def set_coefficient(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the strategy coefficient after unit normalization.

**Parameters**

- `value` (`object`) — Nonnegative scalar coefficient.
- `units` (`object`) — Unit accepted by the concrete strategy builder.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `TypeError` — If ``value`` is not a scalar real value.
- `ValueError` — If ``value`` is invalid or ``units`` is unsupported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L108-L125)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_coefficient_provenance

*method*

```python
def set_coefficient_provenance(value: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the required nonempty coefficient-origin description.

**Parameters**

- `value` (`object`) — Nonempty, non-whitespace provenance string.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `ValueError` — If ``value`` is not a nonempty string.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L127-L142)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_diameter_convention

*method*

```python
def set_diameter_convention(value: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the required ``"mobility_diameter"`` convention.

**Parameters**

- `value` (`object`) — Required convention identifier.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `ValueError` — If ``value`` is not ``"mobility_diameter"``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L311-L326)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_formation_diameter

*method*

```python
def set_formation_diameter(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the positive formation mobility diameter in m.

**Parameters**

- `value` (`object`) — Positive formation diameter.
- `units` (`object`) — ``"m"`` or ``"nm"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L231-L244)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_injection_composition

*method*

```python
def set_injection_composition(value: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Copy and validate future formed-particle composition metadata.

**Parameters**

- `value` (`object`) — Ordered bounded sequence of nonnegative molecule counts with at least one positive count.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `TypeError` — If ``value`` is not an ordered bounded sequence.
- `ValueError` — If the copied composition is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L291-L309)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_parameters

*method*

```python
def set_parameters(parameters: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Atomically configure this builder from its exact mapping schema.

Scalar physical values require matching ``*_units`` entries. Required
values, paired saturation bounds, and all units are validated on a
staging builder before this builder is changed.

**Parameters**

- `parameters` (`object`) — Dictionary containing exactly supported configuration keys and required unit metadata.

**Returns**

- (`Self`) — This builder after successful replacement of its configuration.

**Raises**

- `TypeError` — If ``parameters`` is not a dictionary.
- `ValueError` — If keys, units, required values, or paired bounds are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L376-L414)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_precursor_number_concentration_lower

*method*

```python
def set_precursor_number_concentration_lower(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the lower precursor-number-concentration endpoint in #/m³.

**Parameters**

- `value` (`object`) — Nonnegative endpoint value.
- `units` (`object`) — ``"1/m^3"`` or ``"1/cm^3"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L144-L162)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_precursor_number_concentration_upper

*method*

```python
def set_precursor_number_concentration_upper(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the upper precursor-number-concentration endpoint in #/m³.

**Parameters**

- `value` (`object`) — Nonnegative endpoint value.
- `units` (`object`) — ``"1/m^3"`` or ``"1/cm^3"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L164-L182)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_saturation_lower

*method*

```python
def set_saturation_lower(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the nonnegative lower dimensionless saturation endpoint.

**Parameters**

- `value` (`object`) — Nonnegative saturation endpoint.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L246-L259)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_saturation_upper

*method*

```python
def set_saturation_upper(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the nonnegative upper dimensionless saturation endpoint.

**Parameters**

- `value` (`object`) — Nonnegative saturation endpoint.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L261-L274)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_survival_factor

*method*

```python
def set_survival_factor(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the nonnegative dimensionless survival factor.

**Parameters**

- `value` (`object`) — Nonnegative survival factor.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L276-L289)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_temperature_lower

*method*

```python
def set_temperature_lower(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the positive lower temperature endpoint in K.

**Parameters**

- `value` (`object`) — Positive temperature endpoint.
- `units` (`object`) — Required ``"K"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L201-L214)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.set_temperature_upper

*method*

```python
def set_temperature_upper(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the positive upper temperature endpoint in K.

**Parameters**

- `value` (`object`) — Positive temperature endpoint.
- `units` (`object`) — Required ``"K"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L216-L229)

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.check_keys

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

### particula.dynamics.nucleation.nucleation_builders.ActivationNucleationBuilder.pre_build_check

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

## particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder

*class*

```python
class KineticNucleationBuilder(_NucleationStrategyBuilder)
```

Bases: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Build immutable kinetic potential-rate strategies.

The kinetic coefficient accepts ``"m^3/s"`` or ``"cm^3/s"`` and is
normalized to m³/s before construction.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L501-L529)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.coefficient_units

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_units = {'m^3/s', 'cm^3/s'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L508-L508)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.coefficient

*attribute* · *instance attribute*

```python
coefficient: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L74-L74)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.coefficient_provenance

*attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L75-L75)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.diameter_convention

*attribute* · *instance attribute*

```python
diameter_convention: str | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L85-L85)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.formation_diameter

*attribute* · *instance attribute*

```python
formation_diameter: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L81-L81)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L80-L80)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.precursor_number_concentration_lower

*attribute* · *instance attribute*

```python
precursor_number_concentration_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L76-L76)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.precursor_number_concentration_upper

*attribute* · *instance attribute*

```python
precursor_number_concentration_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L77-L77)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.saturation_lower

*attribute* · *instance attribute*

```python
saturation_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L82-L82)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.saturation_upper

*attribute* · *instance attribute*

```python
saturation_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L83-L83)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.survival_factor

*attribute* · *instance attribute*

```python
survival_factor: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L84-L84)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.temperature_lower

*attribute* · *instance attribute*

```python
temperature_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L78-L78)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.temperature_upper

*attribute* · *instance attribute*

```python
temperature_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L79-L79)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.build

*method*

```python
def build() -> KineticNucleationStrategy
```

Build a fully validated immutable kinetic strategy.

**Returns**

- (`KineticNucleationStrategy`) — Kinetic strategy with normalized metadata and a survival default of
- (`KineticNucleationStrategy`) — ``1.0`` when no survival factor was supplied.

**Raises**

- `ValueError` — If required builder fields or paired saturation bounds
are missing or invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L510-L529)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.__init__

*method*

```python
def __init__() -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Initialize an unconfigured strategy builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L71-L85)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder._configuration

*method*

```python
def _configuration() -> tuple[NucleationValidityDomain, InjectionComposition, FormationMetadata, float]
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L416-L467)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder._converted

*method* · *staticmethod*

```python
def _converted(value: object, unit: object, accepted: set[str], name: str) -> float
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L93-L106)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder._set_mapping_scalar

*method*

```python
def _set_mapping_scalar(key: str, value: object, units: object) -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L353-L374)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder._set_positive

*method*

```python
def _set_positive(name: str, value: object, units: object, accepted: set[str]) -> float
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L184-L199)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder._unit

*method* · *staticmethod*

```python
def _unit(unit: object, accepted: set[str], name: str) -> str
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L87-L91)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder._validate_schema

*method* · *staticmethod*

```python
def _validate_schema(parameters: dict[str, object]) -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L328-L351)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_coefficient

*method*

```python
def set_coefficient(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the strategy coefficient after unit normalization.

**Parameters**

- `value` (`object`) — Nonnegative scalar coefficient.
- `units` (`object`) — Unit accepted by the concrete strategy builder.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `TypeError` — If ``value`` is not a scalar real value.
- `ValueError` — If ``value`` is invalid or ``units`` is unsupported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L108-L125)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_coefficient_provenance

*method*

```python
def set_coefficient_provenance(value: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the required nonempty coefficient-origin description.

**Parameters**

- `value` (`object`) — Nonempty, non-whitespace provenance string.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `ValueError` — If ``value`` is not a nonempty string.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L127-L142)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_diameter_convention

*method*

```python
def set_diameter_convention(value: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the required ``"mobility_diameter"`` convention.

**Parameters**

- `value` (`object`) — Required convention identifier.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `ValueError` — If ``value`` is not ``"mobility_diameter"``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L311-L326)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_formation_diameter

*method*

```python
def set_formation_diameter(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the positive formation mobility diameter in m.

**Parameters**

- `value` (`object`) — Positive formation diameter.
- `units` (`object`) — ``"m"`` or ``"nm"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L231-L244)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_injection_composition

*method*

```python
def set_injection_composition(value: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Copy and validate future formed-particle composition metadata.

**Parameters**

- `value` (`object`) — Ordered bounded sequence of nonnegative molecule counts with at least one positive count.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `TypeError` — If ``value`` is not an ordered bounded sequence.
- `ValueError` — If the copied composition is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L291-L309)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_parameters

*method*

```python
def set_parameters(parameters: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Atomically configure this builder from its exact mapping schema.

Scalar physical values require matching ``*_units`` entries. Required
values, paired saturation bounds, and all units are validated on a
staging builder before this builder is changed.

**Parameters**

- `parameters` (`object`) — Dictionary containing exactly supported configuration keys and required unit metadata.

**Returns**

- (`Self`) — This builder after successful replacement of its configuration.

**Raises**

- `TypeError` — If ``parameters`` is not a dictionary.
- `ValueError` — If keys, units, required values, or paired bounds are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L376-L414)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_precursor_number_concentration_lower

*method*

```python
def set_precursor_number_concentration_lower(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the lower precursor-number-concentration endpoint in #/m³.

**Parameters**

- `value` (`object`) — Nonnegative endpoint value.
- `units` (`object`) — ``"1/m^3"`` or ``"1/cm^3"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L144-L162)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_precursor_number_concentration_upper

*method*

```python
def set_precursor_number_concentration_upper(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the upper precursor-number-concentration endpoint in #/m³.

**Parameters**

- `value` (`object`) — Nonnegative endpoint value.
- `units` (`object`) — ``"1/m^3"`` or ``"1/cm^3"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L164-L182)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_saturation_lower

*method*

```python
def set_saturation_lower(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the nonnegative lower dimensionless saturation endpoint.

**Parameters**

- `value` (`object`) — Nonnegative saturation endpoint.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L246-L259)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_saturation_upper

*method*

```python
def set_saturation_upper(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the nonnegative upper dimensionless saturation endpoint.

**Parameters**

- `value` (`object`) — Nonnegative saturation endpoint.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L261-L274)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_survival_factor

*method*

```python
def set_survival_factor(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the nonnegative dimensionless survival factor.

**Parameters**

- `value` (`object`) — Nonnegative survival factor.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L276-L289)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_temperature_lower

*method*

```python
def set_temperature_lower(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the positive lower temperature endpoint in K.

**Parameters**

- `value` (`object`) — Positive temperature endpoint.
- `units` (`object`) — Required ``"K"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L201-L214)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.set_temperature_upper

*method*

```python
def set_temperature_upper(value: object, units: object) -> Self
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Set the positive upper temperature endpoint in K.

**Parameters**

- `value` (`object`) — Positive temperature endpoint.
- `units` (`object`) — Required ``"K"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L216-L229)

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.check_keys

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

### particula.dynamics.nucleation.nucleation_builders.KineticNucleationBuilder.pre_build_check

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

## particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder

*class*

```python
class NucleationSourceConfigBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Build immutable public construction source-selection metadata.

This builder selects a supported potential-rate strategy and precursor
index only. It does not expose demand planning/source transaction
particle-source finalization or
mutation behavior.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L532-L631)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.precursor_index

*attribute* · *instance attribute*

```python
precursor_index: int | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L547-L547)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.species_map

*attribute* · *instance attribute*

```python
species_map: NucleationSpeciesMap | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L548-L548)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.strategy

*attribute* · *instance attribute*

```python
strategy: NucleationStrategy | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L546-L546)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize an unconfigured public construction source-selection
builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L541-L548)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.build

*method*

```python
def build() -> NucleationSourceConfig
```

Build validated immutable public construction source-selection
    metadata.

**Returns**

- (`NucleationSourceConfig`) — Metadata selecting a supported strategy and precursor index.

**Raises**

- `ValueError` — If required builder fields are missing or invalid.
- `TypeError` — If the precursor index is not an accepted integer.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L615-L631)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.set_parameters

*method*

```python
def set_parameters(parameters: object) -> Self
```

Atomically configure the exact two-key source metadata schema.

**Parameters**

- `parameters` (`object`) — Dictionary containing only ``strategy`` and ``precursor_index``.

**Returns**

- (`Self`) — This builder after a valid configuration is staged and copied.

**Raises**

- `TypeError` — If ``parameters`` is not a dictionary.
- `ValueError` — If keys or source-selection values are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L581-L613)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.set_precursor_index

*method*

```python
def set_precursor_index(precursor_index: object) -> Self
```

Set the candidate nonnegative precursor species index.

**Parameters**

- `precursor_index` (`object`) — Index validated when the configuration is built.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L569-L579)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.set_species_map

*method*

```python
def set_species_map(species_map: NucleationSpeciesMap) -> Self
```

Bind explicit native gas order and injection lane pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L550-L555)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.set_strategy

*method*

```python
def set_strategy(strategy: object) -> Self
```

Set the candidate supported nucleation strategy.

**Parameters**

- `strategy` (`object`) — Activation or kinetic strategy to select.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L557-L567)

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.check_keys

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

### particula.dynamics.nucleation.nucleation_builders.NucleationSourceConfigBuilder.pre_build_check

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

## particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder

*class*

```python
class _NucleationStrategyBuilder(BuilderABC, ABC)
```

Bases: `particula.abc_builder.BuilderABC`, `ABC`

Provide shared strict normalization for potential-rate builders.

Subclasses define their accepted coefficient units and construct the
matching immutable strategy. Required scalar fields are normalized to SI
units before they are retained on the builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L61-L467)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.coefficient

*attribute* · *instance attribute*

```python
coefficient: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L74-L74)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.coefficient_provenance

*attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L75-L75)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.coefficient_units

*attribute* · *instance attribute*

```python
coefficient_units: set[str]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L69-L69)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.diameter_convention

*attribute* · *instance attribute*

```python
diameter_convention: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L85-L85)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.formation_diameter

*attribute* · *instance attribute*

```python
formation_diameter: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L81-L81)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L80-L80)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.precursor_number_concentration_lower

*attribute* · *instance attribute*

```python
precursor_number_concentration_lower: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L76-L76)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.precursor_number_concentration_upper

*attribute* · *instance attribute*

```python
precursor_number_concentration_upper: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L77-L77)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.saturation_lower

*attribute* · *instance attribute*

```python
saturation_lower: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L82-L82)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.saturation_upper

*attribute* · *instance attribute*

```python
saturation_upper: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L83-L83)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.survival_factor

*attribute* · *instance attribute*

```python
survival_factor: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L84-L84)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.temperature_lower

*attribute* · *instance attribute*

```python
temperature_lower: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L78-L78)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.temperature_upper

*attribute* · *instance attribute*

```python
temperature_upper: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L79-L79)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize an unconfigured strategy builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L71-L85)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder._configuration

*method*

```python
def _configuration() -> tuple[NucleationValidityDomain, InjectionComposition, FormationMetadata, float]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L416-L467)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder._converted

*method* · *staticmethod*

```python
def _converted(value: object, unit: object, accepted: set[str], name: str) -> float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L93-L106)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder._set_mapping_scalar

*method*

```python
def _set_mapping_scalar(key: str, value: object, units: object) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L353-L374)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder._set_positive

*method*

```python
def _set_positive(name: str, value: object, units: object, accepted: set[str]) -> float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L184-L199)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder._unit

*method* · *staticmethod*

```python
def _unit(unit: object, accepted: set[str], name: str) -> str
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L87-L91)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder._validate_schema

*method* · *staticmethod*

```python
def _validate_schema(parameters: dict[str, object]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L328-L351)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_coefficient

*method*

```python
def set_coefficient(value: object, units: object) -> Self
```

Set the strategy coefficient after unit normalization.

**Parameters**

- `value` (`object`) — Nonnegative scalar coefficient.
- `units` (`object`) — Unit accepted by the concrete strategy builder.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `TypeError` — If ``value`` is not a scalar real value.
- `ValueError` — If ``value`` is invalid or ``units`` is unsupported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L108-L125)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_coefficient_provenance

*method*

```python
def set_coefficient_provenance(value: object) -> Self
```

Set the required nonempty coefficient-origin description.

**Parameters**

- `value` (`object`) — Nonempty, non-whitespace provenance string.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `ValueError` — If ``value`` is not a nonempty string.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L127-L142)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_diameter_convention

*method*

```python
def set_diameter_convention(value: object) -> Self
```

Set the required ``"mobility_diameter"`` convention.

**Parameters**

- `value` (`object`) — Required convention identifier.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `ValueError` — If ``value`` is not ``"mobility_diameter"``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L311-L326)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_formation_diameter

*method*

```python
def set_formation_diameter(value: object, units: object) -> Self
```

Set the positive formation mobility diameter in m.

**Parameters**

- `value` (`object`) — Positive formation diameter.
- `units` (`object`) — ``"m"`` or ``"nm"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L231-L244)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_injection_composition

*method*

```python
def set_injection_composition(value: object) -> Self
```

Copy and validate future formed-particle composition metadata.

**Parameters**

- `value` (`object`) — Ordered bounded sequence of nonnegative molecule counts with at least one positive count.

**Returns**

- (`Self`) — This builder for fluent configuration.

**Raises**

- `TypeError` — If ``value`` is not an ordered bounded sequence.
- `ValueError` — If the copied composition is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L291-L309)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_parameters

*method*

```python
def set_parameters(parameters: object) -> Self
```

Atomically configure this builder from its exact mapping schema.

Scalar physical values require matching ``*_units`` entries. Required
values, paired saturation bounds, and all units are validated on a
staging builder before this builder is changed.

**Parameters**

- `parameters` (`object`) — Dictionary containing exactly supported configuration keys and required unit metadata.

**Returns**

- (`Self`) — This builder after successful replacement of its configuration.

**Raises**

- `TypeError` — If ``parameters`` is not a dictionary.
- `ValueError` — If keys, units, required values, or paired bounds are
invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L376-L414)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_precursor_number_concentration_lower

*method*

```python
def set_precursor_number_concentration_lower(value: object, units: object) -> Self
```

Set the lower precursor-number-concentration endpoint in #/m³.

**Parameters**

- `value` (`object`) — Nonnegative endpoint value.
- `units` (`object`) — ``"1/m^3"`` or ``"1/cm^3"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L144-L162)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_precursor_number_concentration_upper

*method*

```python
def set_precursor_number_concentration_upper(value: object, units: object) -> Self
```

Set the upper precursor-number-concentration endpoint in #/m³.

**Parameters**

- `value` (`object`) — Nonnegative endpoint value.
- `units` (`object`) — ``"1/m^3"`` or ``"1/cm^3"``.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L164-L182)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_saturation_lower

*method*

```python
def set_saturation_lower(value: object, units: object) -> Self
```

Set the nonnegative lower dimensionless saturation endpoint.

**Parameters**

- `value` (`object`) — Nonnegative saturation endpoint.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L246-L259)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_saturation_upper

*method*

```python
def set_saturation_upper(value: object, units: object) -> Self
```

Set the nonnegative upper dimensionless saturation endpoint.

**Parameters**

- `value` (`object`) — Nonnegative saturation endpoint.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L261-L274)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_survival_factor

*method*

```python
def set_survival_factor(value: object, units: object) -> Self
```

Set the nonnegative dimensionless survival factor.

**Parameters**

- `value` (`object`) — Nonnegative survival factor.
- `units` (`object`) — Required ``"dimensionless"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L276-L289)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_temperature_lower

*method*

```python
def set_temperature_lower(value: object, units: object) -> Self
```

Set the positive lower temperature endpoint in K.

**Parameters**

- `value` (`object`) — Positive temperature endpoint.
- `units` (`object`) — Required ``"K"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L201-L214)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.set_temperature_upper

*method*

```python
def set_temperature_upper(value: object, units: object) -> Self
```

Set the positive upper temperature endpoint in K.

**Parameters**

- `value` (`object`) — Positive temperature endpoint.
- `units` (`object`) — Required ``"K"`` unit.

**Returns**

- (`Self`) — This builder for fluent configuration.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L216-L229)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.build

*method* · *abstract*

```python
def build() -> Any
```

Inherited from: `particula.abc_builder.BuilderABC`

Build and return the strategy object with the set parameters.

**Returns**

- `Any` (`Any`) — The built strategy object.

**Examples**

```py
builder = Builder()
strategy = builder.build()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L153-L165)

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.check_keys

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

### particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder.pre_build_check

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
