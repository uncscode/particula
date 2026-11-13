# particula.dynamics.nucleation

*module*

Expose bounded public construction APIs for CPU nucleation potential rates.

This namespace exports immutable activation and kinetic potential-rate
strategies, their strict builders, ``NucleationFactory``, and source-selection
metadata. Demand planning/source transaction helpers in ``particle_source``
finalize source demand or mutate
particle and gas state; they intentionally remain concrete-module-only.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/__init__.py#L1-L1)

## particula.dynamics.nucleation.ActivationNucleationBuilder

*class*

```python
class ActivationNucleationBuilder(_NucleationStrategyBuilder)
```

Bases: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Re-exported from: `particula.dynamics.nucleation.nucleation_builders`

Build immutable activation potential-rate strategies.

The activation coefficient must use ``"s^-1"`` and all retained physical
values are normalized to SI units before construction.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L470-L498)

### particula.dynamics.nucleation.ActivationNucleationBuilder.coefficient_units

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_units = {'s^-1'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L477-L477)

### particula.dynamics.nucleation.ActivationNucleationBuilder.coefficient

*attribute* · *instance attribute*

```python
coefficient: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L74-L74)

### particula.dynamics.nucleation.ActivationNucleationBuilder.coefficient_provenance

*attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L75-L75)

### particula.dynamics.nucleation.ActivationNucleationBuilder.diameter_convention

*attribute* · *instance attribute*

```python
diameter_convention: str | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L85-L85)

### particula.dynamics.nucleation.ActivationNucleationBuilder.formation_diameter

*attribute* · *instance attribute*

```python
formation_diameter: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L81-L81)

### particula.dynamics.nucleation.ActivationNucleationBuilder.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L80-L80)

### particula.dynamics.nucleation.ActivationNucleationBuilder.precursor_number_concentration_lower

*attribute* · *instance attribute*

```python
precursor_number_concentration_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L76-L76)

### particula.dynamics.nucleation.ActivationNucleationBuilder.precursor_number_concentration_upper

*attribute* · *instance attribute*

```python
precursor_number_concentration_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L77-L77)

### particula.dynamics.nucleation.ActivationNucleationBuilder.saturation_lower

*attribute* · *instance attribute*

```python
saturation_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L82-L82)

### particula.dynamics.nucleation.ActivationNucleationBuilder.saturation_upper

*attribute* · *instance attribute*

```python
saturation_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L83-L83)

### particula.dynamics.nucleation.ActivationNucleationBuilder.survival_factor

*attribute* · *instance attribute*

```python
survival_factor: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L84-L84)

### particula.dynamics.nucleation.ActivationNucleationBuilder.temperature_lower

*attribute* · *instance attribute*

```python
temperature_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L78-L78)

### particula.dynamics.nucleation.ActivationNucleationBuilder.temperature_upper

*attribute* · *instance attribute*

```python
temperature_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L79-L79)

### particula.dynamics.nucleation.ActivationNucleationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.nucleation.ActivationNucleationBuilder.build

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.__init__

*method*

```python
def __init__() -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Initialize an unconfigured strategy builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L71-L85)

### particula.dynamics.nucleation.ActivationNucleationBuilder._configuration

*method*

```python
def _configuration() -> tuple[NucleationValidityDomain, InjectionComposition, FormationMetadata, float]
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L416-L467)

### particula.dynamics.nucleation.ActivationNucleationBuilder._converted

*method* · *staticmethod*

```python
def _converted(value: object, unit: object, accepted: set[str], name: str) -> float
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L93-L106)

### particula.dynamics.nucleation.ActivationNucleationBuilder._set_mapping_scalar

*method*

```python
def _set_mapping_scalar(key: str, value: object, units: object) -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L353-L374)

### particula.dynamics.nucleation.ActivationNucleationBuilder._set_positive

*method*

```python
def _set_positive(name: str, value: object, units: object, accepted: set[str]) -> float
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L184-L199)

### particula.dynamics.nucleation.ActivationNucleationBuilder._unit

*method* · *staticmethod*

```python
def _unit(unit: object, accepted: set[str], name: str) -> str
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L87-L91)

### particula.dynamics.nucleation.ActivationNucleationBuilder._validate_schema

*method* · *staticmethod*

```python
def _validate_schema(parameters: dict[str, object]) -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L328-L351)

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_coefficient

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_coefficient_provenance

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_diameter_convention

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_formation_diameter

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_injection_composition

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_parameters

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_precursor_number_concentration_lower

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_precursor_number_concentration_upper

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_saturation_lower

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_saturation_upper

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_survival_factor

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_temperature_lower

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.set_temperature_upper

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.check_keys

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

### particula.dynamics.nucleation.ActivationNucleationBuilder.pre_build_check

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

## particula.dynamics.nucleation.ActivationNucleationStrategy

*class* · *dataclass*

```python
class ActivationNucleationStrategy(NucleationStrategy)
```

Bases: `particula.dynamics.nucleation.nucleation_strategies.NucleationStrategy`

Re-exported from: `particula.dynamics.nucleation.nucleation_strategies`

Activation potential-rate parameterization, ``J = A × C × S``.

``A`` is the nonnegative activation coefficient [s⁻¹], ``C`` is precursor
number concentration [#/m³], and ``S`` is a caller-supplied dimensionless
survival factor. A valid zero coefficient, concentration, or survival
factor returns exactly ``0.0`` after basic input validation and conversion.

**Attributes**

- `coefficient` (`float`) — Activation coefficient [s⁻¹].
- `validity_domain` (`NucleationValidityDomain`) — Inclusive parameterization domain.
- `injection_composition` (`InjectionComposition`) — Metadata for a future formed particle.
- `formation_metadata` (`FormationMetadata`) — Potential-rate formation-size metadata.
- `survival_factor` (`float`) — Caller-supplied nonnegative dimensionless factor.
- `coefficient_provenance` (`str | None`) — Optional description of coefficient origin.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L515-L611)

### particula.dynamics.nucleation.ActivationNucleationStrategy.coefficient

*attribute* · *instance attribute*

```python
coefficient: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L533-L533)

### particula.dynamics.nucleation.ActivationNucleationStrategy.coefficient_provenance

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L538-L538)

### particula.dynamics.nucleation.ActivationNucleationStrategy.formation_metadata

*attribute* · *instance attribute*

```python
formation_metadata: FormationMetadata
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L536-L536)

### particula.dynamics.nucleation.ActivationNucleationStrategy.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L535-L535)

### particula.dynamics.nucleation.ActivationNucleationStrategy.survival_factor

*attribute* · *class attribute* · *instance attribute*

```python
survival_factor: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L537-L537)

### particula.dynamics.nucleation.ActivationNucleationStrategy.validity_domain

*attribute* · *instance attribute*

```python
validity_domain: NucleationValidityDomain
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L534-L534)

### particula.dynamics.nucleation.ActivationNucleationStrategy.__init__

*method*

```python
def __init__(coefficient: float, validity_domain: NucleationValidityDomain, injection_composition: InjectionComposition, formation_metadata: FormationMetadata, survival_factor: float = 1.0, coefficient_provenance: str | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.ActivationNucleationStrategy.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate immutable activation-strategy configuration.

**Raises**

- `TypeError` — If a scalar configuration value is not a scalar real.
- `ValueError` — If configuration values or metadata records are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L540-L554)

### particula.dynamics.nucleation.ActivationNucleationStrategy.potential_rate

*method*

```python
def potential_rate(precursor_mass_concentration: float, precursor_molar_mass: float, temperature: float, saturation: float | None = None) -> float
```

Calculate the activation potential event rate [#/m³/s].

Basic inputs and number-concentration conversion are validated before
exact zero paths. Nonzero inputs must lie in the configured inclusive
concentration and temperature intervals; below-gate saturation returns
exactly ``0.0``.

**Parameters**

- `precursor_mass_concentration` (`float`) — Precursor mass concentration [kg/m³].
- `precursor_molar_mass` (`float`) — Precursor molar mass [kg/mol].
- `temperature` (`float`) — Temperature [K].
- `saturation` (`float | None`) (default: `None`) — Dimensionless saturation when configured, else ``None``.

**Returns**

- (`float`) — Potential activation event rate [#/m³/s].

**Raises**

- `TypeError` — If required inputs are not accepted scalar real values.
- `ValueError` — If inputs are physically invalid, out of domain, or
yield a nonfinite conversion or rate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L556-L611)

## particula.dynamics.nucleation.ClosedInterval

*class* · *dataclass*

```python
class ClosedInterval
```

Re-exported from: `particula.dynamics.nucleation.nucleation_strategies`

Finite inclusive interval for a scalar physical quantity.

**Attributes**

- `lower` (`float`) — Finite inclusive lower endpoint.
- `upper` (`float`) — Finite inclusive upper endpoint, no less than ``lower``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L113-L146)

### particula.dynamics.nucleation.ClosedInterval.lower

*attribute* · *instance attribute*

```python
lower: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L122-L122)

### particula.dynamics.nucleation.ClosedInterval.upper

*attribute* · *instance attribute*

```python
upper: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L123-L123)

### particula.dynamics.nucleation.ClosedInterval.__init__

*method*

```python
def __init__(lower: float, upper: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.ClosedInterval.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate finite ordered interval endpoints.

**Raises**

- `TypeError` — If an endpoint is not a scalar real value.
- `ValueError` — If an endpoint is nonfinite or ``lower > upper``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L125-L135)

### particula.dynamics.nucleation.ClosedInterval.contains

*method*

```python
def contains(value: float) -> bool
```

Test inclusive membership for an already validated value.

**Parameters**

- `value` (`float`) — Normalized scalar value to test.

**Returns**

- (`bool`) — ``True`` if ``lower <= value <= upper``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L137-L146)

## particula.dynamics.nucleation.FormationMetadata

*class* · *dataclass*

```python
class FormationMetadata
```

Re-exported from: `particula.dynamics.nucleation.nucleation_strategies`

Metadata for the size represented by a potential formation rate.

**Attributes**

- `formation_diameter` (`float`) — Positive mobility diameter [m].
- `diameter_convention` (`str`) — Required ``"mobility_diameter"`` convention.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L218-L239)

### particula.dynamics.nucleation.FormationMetadata.diameter_convention

*attribute* · *class attribute* · *instance attribute*

```python
diameter_convention: str = 'mobility_diameter'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L228-L228)

### particula.dynamics.nucleation.FormationMetadata.formation_diameter

*attribute* · *instance attribute*

```python
formation_diameter: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L227-L227)

### particula.dynamics.nucleation.FormationMetadata.__init__

*method*

```python
def __init__(formation_diameter: float, diameter_convention: str = 'mobility_diameter') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.FormationMetadata.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the potential-rate size representation.

**Raises**

- `TypeError` — If the diameter is not a scalar real value.
- `ValueError` — If the diameter is invalid or its convention differs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L230-L239)

## particula.dynamics.nucleation.InjectionComposition

*class* · *dataclass*

```python
class InjectionComposition
```

Re-exported from: `particula.dynamics.nucleation.nucleation_strategies`

Future species-order molecule counts for one formed particle.

The records preserve composition metadata but do not inject a particle in
this potential-rate-only implementation.

**Attributes**

- `molecule_counts` (`tuple[int, ...]`) — Nonempty molecule counts by future species order.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L182-L215)

### particula.dynamics.nucleation.InjectionComposition.molecule_counts

*attribute* · *instance attribute*

```python
molecule_counts: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L193-L193)

### particula.dynamics.nucleation.InjectionComposition.__init__

*method*

```python
def __init__(molecule_counts: tuple[int, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.InjectionComposition.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the nonempty, nonnegative injection composition.

**Raises**

- `TypeError` — If counts are not a tuple of non-Boolean integers.
- `ValueError` — If no count is positive or any count is negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L195-L215)

## particula.dynamics.nucleation.KineticNucleationBuilder

*class*

```python
class KineticNucleationBuilder(_NucleationStrategyBuilder)
```

Bases: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Re-exported from: `particula.dynamics.nucleation.nucleation_builders`

Build immutable kinetic potential-rate strategies.

The kinetic coefficient accepts ``"m^3/s"`` or ``"cm^3/s"`` and is
normalized to m³/s before construction.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L501-L529)

### particula.dynamics.nucleation.KineticNucleationBuilder.coefficient_units

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_units = {'m^3/s', 'cm^3/s'}
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L508-L508)

### particula.dynamics.nucleation.KineticNucleationBuilder.coefficient

*attribute* · *instance attribute*

```python
coefficient: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L74-L74)

### particula.dynamics.nucleation.KineticNucleationBuilder.coefficient_provenance

*attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L75-L75)

### particula.dynamics.nucleation.KineticNucleationBuilder.diameter_convention

*attribute* · *instance attribute*

```python
diameter_convention: str | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L85-L85)

### particula.dynamics.nucleation.KineticNucleationBuilder.formation_diameter

*attribute* · *instance attribute*

```python
formation_diameter: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L81-L81)

### particula.dynamics.nucleation.KineticNucleationBuilder.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L80-L80)

### particula.dynamics.nucleation.KineticNucleationBuilder.precursor_number_concentration_lower

*attribute* · *instance attribute*

```python
precursor_number_concentration_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L76-L76)

### particula.dynamics.nucleation.KineticNucleationBuilder.precursor_number_concentration_upper

*attribute* · *instance attribute*

```python
precursor_number_concentration_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L77-L77)

### particula.dynamics.nucleation.KineticNucleationBuilder.saturation_lower

*attribute* · *instance attribute*

```python
saturation_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L82-L82)

### particula.dynamics.nucleation.KineticNucleationBuilder.saturation_upper

*attribute* · *instance attribute*

```python
saturation_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L83-L83)

### particula.dynamics.nucleation.KineticNucleationBuilder.survival_factor

*attribute* · *instance attribute*

```python
survival_factor: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L84-L84)

### particula.dynamics.nucleation.KineticNucleationBuilder.temperature_lower

*attribute* · *instance attribute*

```python
temperature_lower: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L78-L78)

### particula.dynamics.nucleation.KineticNucleationBuilder.temperature_upper

*attribute* · *instance attribute*

```python
temperature_upper: float | None = None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L79-L79)

### particula.dynamics.nucleation.KineticNucleationBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.nucleation.KineticNucleationBuilder.build

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

### particula.dynamics.nucleation.KineticNucleationBuilder.__init__

*method*

```python
def __init__() -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

Initialize an unconfigured strategy builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L71-L85)

### particula.dynamics.nucleation.KineticNucleationBuilder._configuration

*method*

```python
def _configuration() -> tuple[NucleationValidityDomain, InjectionComposition, FormationMetadata, float]
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L416-L467)

### particula.dynamics.nucleation.KineticNucleationBuilder._converted

*method* · *staticmethod*

```python
def _converted(value: object, unit: object, accepted: set[str], name: str) -> float
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L93-L106)

### particula.dynamics.nucleation.KineticNucleationBuilder._set_mapping_scalar

*method*

```python
def _set_mapping_scalar(key: str, value: object, units: object) -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L353-L374)

### particula.dynamics.nucleation.KineticNucleationBuilder._set_positive

*method*

```python
def _set_positive(name: str, value: object, units: object, accepted: set[str]) -> float
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L184-L199)

### particula.dynamics.nucleation.KineticNucleationBuilder._unit

*method* · *staticmethod*

```python
def _unit(unit: object, accepted: set[str], name: str) -> str
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L87-L91)

### particula.dynamics.nucleation.KineticNucleationBuilder._validate_schema

*method* · *staticmethod*

```python
def _validate_schema(parameters: dict[str, object]) -> None
```

Inherited from: `particula.dynamics.nucleation.nucleation_builders._NucleationStrategyBuilder`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L328-L351)

### particula.dynamics.nucleation.KineticNucleationBuilder.set_coefficient

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_coefficient_provenance

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_diameter_convention

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_formation_diameter

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_injection_composition

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_parameters

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_precursor_number_concentration_lower

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_precursor_number_concentration_upper

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_saturation_lower

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_saturation_upper

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_survival_factor

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_temperature_lower

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

### particula.dynamics.nucleation.KineticNucleationBuilder.set_temperature_upper

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

### particula.dynamics.nucleation.KineticNucleationBuilder.check_keys

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

### particula.dynamics.nucleation.KineticNucleationBuilder.pre_build_check

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

## particula.dynamics.nucleation.KineticNucleationStrategy

*class* · *dataclass*

```python
class KineticNucleationStrategy(NucleationStrategy)
```

Bases: `particula.dynamics.nucleation.nucleation_strategies.NucleationStrategy`

Re-exported from: `particula.dynamics.nucleation.nucleation_strategies`

Kinetic potential-rate parameterization, ``J = K × C² × S``.

``K`` is the nonnegative kinetic coefficient [m³/s], ``C`` is precursor
number concentration [#/m³], and ``S`` is a caller-supplied dimensionless
survival factor. A valid zero coefficient, concentration, or survival
factor returns exactly ``0.0`` after basic input validation and conversion.

**Attributes**

- `coefficient` (`float`) — Kinetic coefficient [m³/s].
- `validity_domain` (`NucleationValidityDomain`) — Inclusive parameterization domain.
- `injection_composition` (`InjectionComposition`) — Metadata for a future formed particle.
- `formation_metadata` (`FormationMetadata`) — Potential-rate formation-size metadata.
- `survival_factor` (`float`) — Caller-supplied nonnegative dimensionless factor.
- `coefficient_provenance` (`str | None`) — Optional description of coefficient origin.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L614-L710)

### particula.dynamics.nucleation.KineticNucleationStrategy.coefficient

*attribute* · *instance attribute*

```python
coefficient: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L632-L632)

### particula.dynamics.nucleation.KineticNucleationStrategy.coefficient_provenance

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L637-L637)

### particula.dynamics.nucleation.KineticNucleationStrategy.formation_metadata

*attribute* · *instance attribute*

```python
formation_metadata: FormationMetadata
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L635-L635)

### particula.dynamics.nucleation.KineticNucleationStrategy.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L634-L634)

### particula.dynamics.nucleation.KineticNucleationStrategy.survival_factor

*attribute* · *class attribute* · *instance attribute*

```python
survival_factor: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L636-L636)

### particula.dynamics.nucleation.KineticNucleationStrategy.validity_domain

*attribute* · *instance attribute*

```python
validity_domain: NucleationValidityDomain
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L633-L633)

### particula.dynamics.nucleation.KineticNucleationStrategy.__init__

*method*

```python
def __init__(coefficient: float, validity_domain: NucleationValidityDomain, injection_composition: InjectionComposition, formation_metadata: FormationMetadata, survival_factor: float = 1.0, coefficient_provenance: str | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.KineticNucleationStrategy.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate immutable kinetic-strategy configuration.

**Raises**

- `TypeError` — If a scalar configuration value is not a scalar real.
- `ValueError` — If configuration values or metadata records are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L639-L653)

### particula.dynamics.nucleation.KineticNucleationStrategy.potential_rate

*method*

```python
def potential_rate(precursor_mass_concentration: float, precursor_molar_mass: float, temperature: float, saturation: float | None = None) -> float
```

Calculate the kinetic potential event rate [#/m³/s].

Basic inputs and number-concentration conversion are validated before
exact zero paths. Nonzero inputs must lie in the configured inclusive
concentration and temperature intervals; below-gate saturation returns
exactly ``0.0``.

**Parameters**

- `precursor_mass_concentration` (`float`) — Precursor mass concentration [kg/m³].
- `precursor_molar_mass` (`float`) — Precursor molar mass [kg/mol].
- `temperature` (`float`) — Temperature [K].
- `saturation` (`float | None`) (default: `None`) — Dimensionless saturation when configured, else ``None``.

**Returns**

- (`float`) — Potential kinetic event rate [#/m³/s].

**Raises**

- `TypeError` — If required inputs are not accepted scalar real values.
- `ValueError` — If inputs are physically invalid, out of domain, or
yield a nonfinite conversion or rate.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L655-L710)

## particula.dynamics.nucleation.NucleationFactory

*class*

```python
class NucleationFactory(StrategyFactoryABC[ActivationNucleationBuilder | KineticNucleationBuilder, ActivationNucleationStrategy | KineticNucleationStrategy])
```

Bases: ``

Re-exported from: `particula.dynamics.nucleation.nucleation_factories`

Build activation or kinetic public construction potential-rate strategies
    by name.

Each request gets a new builder, so failed or successful requests cannot
retain configuration into subsequent calls. Identifiers are case-insensitive
but are limited to ``"activation"`` and ``"kinetic"``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_factories.py#L21-L76)

### particula.dynamics.nucleation.NucleationFactory.get_builders

*method*

```python
def get_builders() -> dict[str, ActivationNucleationBuilder | KineticNucleationBuilder]
```

Return fresh builders for the two supported identifiers.

**Returns**

- (`dict[str, ActivationNucleationBuilder | KineticNucleationBuilder]`) — New activation and kinetic builders under their canonical lowercase
- (`dict[str, ActivationNucleationBuilder | KineticNucleationBuilder]`) — identifiers.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_factories.py#L35-L47)

### particula.dynamics.nucleation.NucleationFactory.get_strategy

*method*

```python
def get_strategy(strategy_type: str, parameters: dict[str, object] | None = None) -> ActivationNucleationStrategy | KineticNucleationStrategy
```

Build a validated strategy from a copied strict-schema payload.

**Parameters**

- `strategy_type` (`str`) — Case-insensitive ``"activation"`` or ``"kinetic"`` identifier.
- `parameters` (`dict[str, object] | None`) (default: `None`) — Required dictionary accepted by the selected builder.

**Returns**

- (`ActivationNucleationStrategy | KineticNucleationStrategy`) — Newly built immutable activation or kinetic potential-rate strategy.

**Raises**

- `ValueError` — If the identifier or payload is invalid, unsupported, or
fails the selected builder's schema validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_factories.py#L49-L76)

## particula.dynamics.nucleation.NucleationSourceConfig

*class* · *dataclass*

```python
class NucleationSourceConfig
```

Re-exported from: `particula.dynamics.nucleation.nucleation_configuration`

Select a public construction nucleation strategy and its precursor
    species index.

This record is metadata only; source-demand finalization and mutation remain
at concrete demand planning/source transaction boundaries.

**Attributes**

- `strategy` (`NucleationStrategy`) — Immutable activation or kinetic potential-rate strategy.
- `precursor_index` (`int`) — Nonnegative index of the precursor species selected for a later concrete source operation.
- `species_map` (`NucleationSpeciesMap | None`) — Explicit native gas order and injection mapping; None is retained only for transitional legacy callers.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L21-L65)

### particula.dynamics.nucleation.NucleationSourceConfig.precursor_index

*attribute* · *instance attribute*

```python
precursor_index: int
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L38-L38)

### particula.dynamics.nucleation.NucleationSourceConfig.species_map

*attribute* · *class attribute* · *instance attribute*

```python
species_map: NucleationSpeciesMap | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L39-L39)

### particula.dynamics.nucleation.NucleationSourceConfig.strategy

*attribute* · *instance attribute*

```python
strategy: NucleationStrategy
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L37-L37)

### particula.dynamics.nucleation.NucleationSourceConfig.__init__

*method*

```python
def __init__(strategy: NucleationStrategy, precursor_index: int, species_map: NucleationSpeciesMap | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L0-L0)

### particula.dynamics.nucleation.NucleationSourceConfig.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the supported strategy types and nonnegative index.

**Raises**

- `TypeError` — If ``precursor_index`` is not a non-Boolean integer.
- `ValueError` — If the strategy is unsupported or the index is negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_configuration.py#L41-L65)

## particula.dynamics.nucleation.NucleationSourceConfigBuilder

*class*

```python
class NucleationSourceConfigBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.dynamics.nucleation.nucleation_builders`

Build immutable public construction source-selection metadata.

This builder selects a supported potential-rate strategy and precursor
index only. It does not expose demand planning/source transaction
particle-source finalization or
mutation behavior.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L532-L631)

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.precursor_index

*attribute* · *instance attribute*

```python
precursor_index: int | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L547-L547)

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.species_map

*attribute* · *instance attribute*

```python
species_map: NucleationSpeciesMap | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L548-L548)

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.strategy

*attribute* · *instance attribute*

```python
strategy: NucleationStrategy | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L546-L546)

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize an unconfigured public construction source-selection
builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L541-L548)

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.build

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

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.set_parameters

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

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.set_precursor_index

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

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.set_species_map

*method*

```python
def set_species_map(species_map: NucleationSpeciesMap) -> Self
```

Bind explicit native gas order and injection lane pairs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_builders.py#L550-L555)

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.set_strategy

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

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.check_keys

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

### particula.dynamics.nucleation.NucleationSourceConfigBuilder.pre_build_check

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

## particula.dynamics.nucleation.NucleationStrategy

*class*

```python
class NucleationStrategy(ABC)
```

Bases: `ABC`

Re-exported from: `particula.dynamics.nucleation.nucleation_strategies`

Abstract interface for scalar nucleation potential-rate strategies.

Implementations return a potential event rate only; they do not represent
particle sources, survival inference, or state mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L242-L267)

### particula.dynamics.nucleation.NucleationStrategy.potential_rate

*method* · *abstract*

```python
def potential_rate(precursor_mass_concentration: float, precursor_molar_mass: float, temperature: float, saturation: float | None = None) -> float
```

Calculate a potential formation-event rate [#/m³/s].

**Parameters**

- `precursor_mass_concentration` (`float`) — Precursor mass concentration [kg/m³].
- `precursor_molar_mass` (`float`) — Precursor molar mass [kg/mol].
- `temperature` (`float`) — Temperature [K].
- `saturation` (`float | None`) (default: `None`) — Optional dimensionless saturation.

**Returns**

- (`float`) — Potential formation-event rate [#/m³/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L249-L267)

## particula.dynamics.nucleation.NucleationValidityDomain

*class* · *dataclass*

```python
class NucleationValidityDomain
```

Re-exported from: `particula.dynamics.nucleation.nucleation_strategies`

Inclusive physical domain for a nucleation-rate parameterization.

**Attributes**

- `precursor_number_concentration` (`ClosedInterval`) — Allowed precursor concentration [#/m³].
- `temperature` (`ClosedInterval`) — Allowed temperature [K].
- `saturation` (`ClosedInterval | None`) — Optional allowed dimensionless saturation interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L149-L179)

### particula.dynamics.nucleation.NucleationValidityDomain.precursor_number_concentration

*attribute* · *instance attribute*

```python
precursor_number_concentration: ClosedInterval
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L159-L159)

### particula.dynamics.nucleation.NucleationValidityDomain.saturation

*attribute* · *class attribute* · *instance attribute*

```python
saturation: ClosedInterval | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L161-L161)

### particula.dynamics.nucleation.NucleationValidityDomain.temperature

*attribute* · *instance attribute*

```python
temperature: ClosedInterval
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L160-L160)

### particula.dynamics.nucleation.NucleationValidityDomain.__init__

*method*

```python
def __init__(precursor_number_concentration: ClosedInterval, temperature: ClosedInterval, saturation: ClosedInterval | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.NucleationValidityDomain.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the nested interval records.

**Raises**

- `ValueError` — If a required field is not a ``ClosedInterval`` or an
optional saturation field is neither one nor ``None``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L163-L179)

**Modules**

- `particula.dynamics.nucleation.nucleation_builders`
- `particula.dynamics.nucleation.nucleation_configuration`
- `particula.dynamics.nucleation.nucleation_factories`
- `particula.dynamics.nucleation.nucleation_strategies`
- `particula.dynamics.nucleation.particle_source`
- `particula.dynamics.nucleation.species_mapping`
