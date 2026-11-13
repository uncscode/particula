# particula.dynamics.nucleation.nucleation_strategies

*module*

Validated CPU-only potential-rate strategies for particle nucleation.

Activation and kinetic parameterizations accept precursor mass concentration
[kg/m³], convert it to number concentration [#/m³], and return potential rates
[#/m³/s]. They do not create particles or mutate gas, particle, or slot state.
Direct potential-rate evaluation construction remains supported. The
deliberately bounded public construction
construction API is exported through ``particula.dynamics.nucleation``;
concrete demand planning/source transaction ``particle_source`` finalization and
state mutation are not.

Kulmala, M., et al. (2006). Toward direct measurement of atmospheric
nucleation. *Science*, 318, 89--92.
https://doi.org/10.1126/science.1144124

Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry and Physics*.
Kerminen, V.-M., & Kulmala, M. (2002). Analytical formulae connecting the
"real" and the "apparent" nucleation rate. *Journal of Aerosol Science*, 33,
609--622. https://doi.org/10.1016/S0021-8502(01)00194-X

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L1-L1)

## particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy

*class* · *dataclass*

```python
class ActivationNucleationStrategy(NucleationStrategy)
```

Bases: `particula.dynamics.nucleation.nucleation_strategies.NucleationStrategy`

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

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.coefficient

*attribute* · *instance attribute*

```python
coefficient: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L533-L533)

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.coefficient_provenance

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L538-L538)

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.formation_metadata

*attribute* · *instance attribute*

```python
formation_metadata: FormationMetadata
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L536-L536)

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L535-L535)

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.survival_factor

*attribute* · *class attribute* · *instance attribute*

```python
survival_factor: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L537-L537)

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.validity_domain

*attribute* · *instance attribute*

```python
validity_domain: NucleationValidityDomain
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L534-L534)

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.__init__

*method*

```python
def __init__(coefficient: float, validity_domain: NucleationValidityDomain, injection_composition: InjectionComposition, formation_metadata: FormationMetadata, survival_factor: float = 1.0, coefficient_provenance: str | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate immutable activation-strategy configuration.

**Raises**

- `TypeError` — If a scalar configuration value is not a scalar real.
- `ValueError` — If configuration values or metadata records are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L540-L554)

### particula.dynamics.nucleation.nucleation_strategies.ActivationNucleationStrategy.potential_rate

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

## particula.dynamics.nucleation.nucleation_strategies.ClosedInterval

*class* · *dataclass*

```python
class ClosedInterval
```

Finite inclusive interval for a scalar physical quantity.

**Attributes**

- `lower` (`float`) — Finite inclusive lower endpoint.
- `upper` (`float`) — Finite inclusive upper endpoint, no less than ``lower``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L113-L146)

### particula.dynamics.nucleation.nucleation_strategies.ClosedInterval.lower

*attribute* · *instance attribute*

```python
lower: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L122-L122)

### particula.dynamics.nucleation.nucleation_strategies.ClosedInterval.upper

*attribute* · *instance attribute*

```python
upper: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L123-L123)

### particula.dynamics.nucleation.nucleation_strategies.ClosedInterval.__init__

*method*

```python
def __init__(lower: float, upper: float) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.nucleation_strategies.ClosedInterval.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate finite ordered interval endpoints.

**Raises**

- `TypeError` — If an endpoint is not a scalar real value.
- `ValueError` — If an endpoint is nonfinite or ``lower > upper``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L125-L135)

### particula.dynamics.nucleation.nucleation_strategies.ClosedInterval.contains

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

## particula.dynamics.nucleation.nucleation_strategies.FormationMetadata

*class* · *dataclass*

```python
class FormationMetadata
```

Metadata for the size represented by a potential formation rate.

**Attributes**

- `formation_diameter` (`float`) — Positive mobility diameter [m].
- `diameter_convention` (`str`) — Required ``"mobility_diameter"`` convention.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L218-L239)

### particula.dynamics.nucleation.nucleation_strategies.FormationMetadata.diameter_convention

*attribute* · *class attribute* · *instance attribute*

```python
diameter_convention: str = 'mobility_diameter'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L228-L228)

### particula.dynamics.nucleation.nucleation_strategies.FormationMetadata.formation_diameter

*attribute* · *instance attribute*

```python
formation_diameter: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L227-L227)

### particula.dynamics.nucleation.nucleation_strategies.FormationMetadata.__init__

*method*

```python
def __init__(formation_diameter: float, diameter_convention: str = 'mobility_diameter') -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.nucleation_strategies.FormationMetadata.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the potential-rate size representation.

**Raises**

- `TypeError` — If the diameter is not a scalar real value.
- `ValueError` — If the diameter is invalid or its convention differs.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L230-L239)

## particula.dynamics.nucleation.nucleation_strategies.InjectionComposition

*class* · *dataclass*

```python
class InjectionComposition
```

Future species-order molecule counts for one formed particle.

The records preserve composition metadata but do not inject a particle in
this potential-rate-only implementation.

**Attributes**

- `molecule_counts` (`tuple[int, ...]`) — Nonempty molecule counts by future species order.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L182-L215)

### particula.dynamics.nucleation.nucleation_strategies.InjectionComposition.molecule_counts

*attribute* · *instance attribute*

```python
molecule_counts: tuple[int, ...]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L193-L193)

### particula.dynamics.nucleation.nucleation_strategies.InjectionComposition.__init__

*method*

```python
def __init__(molecule_counts: tuple[int, ...]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.nucleation_strategies.InjectionComposition.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the nonempty, nonnegative injection composition.

**Raises**

- `TypeError` — If counts are not a tuple of non-Boolean integers.
- `ValueError` — If no count is positive or any count is negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L195-L215)

## particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy

*class* · *dataclass*

```python
class KineticNucleationStrategy(NucleationStrategy)
```

Bases: `particula.dynamics.nucleation.nucleation_strategies.NucleationStrategy`

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

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.coefficient

*attribute* · *instance attribute*

```python
coefficient: float
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L632-L632)

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.coefficient_provenance

*attribute* · *class attribute* · *instance attribute*

```python
coefficient_provenance: str | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L637-L637)

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.formation_metadata

*attribute* · *instance attribute*

```python
formation_metadata: FormationMetadata
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L635-L635)

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.injection_composition

*attribute* · *instance attribute*

```python
injection_composition: InjectionComposition
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L634-L634)

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.survival_factor

*attribute* · *class attribute* · *instance attribute*

```python
survival_factor: float = 1.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L636-L636)

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.validity_domain

*attribute* · *instance attribute*

```python
validity_domain: NucleationValidityDomain
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L633-L633)

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.__init__

*method*

```python
def __init__(coefficient: float, validity_domain: NucleationValidityDomain, injection_composition: InjectionComposition, formation_metadata: FormationMetadata, survival_factor: float = 1.0, coefficient_provenance: str | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate immutable kinetic-strategy configuration.

**Raises**

- `TypeError` — If a scalar configuration value is not a scalar real.
- `ValueError` — If configuration values or metadata records are invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L639-L653)

### particula.dynamics.nucleation.nucleation_strategies.KineticNucleationStrategy.potential_rate

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

## particula.dynamics.nucleation.nucleation_strategies.NucleationStrategy

*class*

```python
class NucleationStrategy(ABC)
```

Bases: `ABC`

Abstract interface for scalar nucleation potential-rate strategies.

Implementations return a potential event rate only; they do not represent
particle sources, survival inference, or state mutation.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L242-L267)

### particula.dynamics.nucleation.nucleation_strategies.NucleationStrategy.potential_rate

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

## particula.dynamics.nucleation.nucleation_strategies.NucleationValidityDomain

*class* · *dataclass*

```python
class NucleationValidityDomain
```

Inclusive physical domain for a nucleation-rate parameterization.

**Attributes**

- `precursor_number_concentration` (`ClosedInterval`) — Allowed precursor concentration [#/m³].
- `temperature` (`ClosedInterval`) — Allowed temperature [K].
- `saturation` (`ClosedInterval | None`) — Optional allowed dimensionless saturation interval.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L149-L179)

### particula.dynamics.nucleation.nucleation_strategies.NucleationValidityDomain.precursor_number_concentration

*attribute* · *instance attribute*

```python
precursor_number_concentration: ClosedInterval
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L159-L159)

### particula.dynamics.nucleation.nucleation_strategies.NucleationValidityDomain.saturation

*attribute* · *class attribute* · *instance attribute*

```python
saturation: ClosedInterval | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L161-L161)

### particula.dynamics.nucleation.nucleation_strategies.NucleationValidityDomain.temperature

*attribute* · *instance attribute*

```python
temperature: ClosedInterval
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L160-L160)

### particula.dynamics.nucleation.nucleation_strategies.NucleationValidityDomain.__init__

*method*

```python
def __init__(precursor_number_concentration: ClosedInterval, temperature: ClosedInterval, saturation: ClosedInterval | None = None) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L0-L0)

### particula.dynamics.nucleation.nucleation_strategies.NucleationValidityDomain.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the nested interval records.

**Raises**

- `ValueError` — If a required field is not a ``ClosedInterval`` or an
optional saturation field is neither one nor ``None``.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L163-L179)

## particula.dynamics.nucleation.nucleation_strategies._calculate_rate

*function*

```python
def _calculate_rate(coefficient: float, concentration: float, survival_factor: float, kinetic: bool) -> float
```

Calculate a finite activation or kinetic potential rate [#/m³/s].

**Parameters**

- `coefficient` (`float`) — Validated activation [s⁻¹] or kinetic [m³/s] coefficient.
- `concentration` (`float`) — Validated precursor number concentration [#/m³].
- `survival_factor` (`float`) — Validated dimensionless survival factor.
- `kinetic` (`bool`) — Whether to use the kinetic ``C²`` relation instead of ``C``.

**Returns**

- (`float`) — Finite potential formation-event rate [#/m³/s].

**Raises**

- `ValueError` — If the final float64 rate is nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L474-L512)

## particula.dynamics.nucleation.nucleation_strategies._convert_number_concentration

*function*

```python
def _convert_number_concentration(precursor_mass_concentration: float, precursor_molar_mass: float) -> float
```

Convert precursor mass concentration to number concentration [#/m³].

**Parameters**

- `precursor_mass_concentration` (`float`) — Validated precursor mass concentration [kg/m³].
- `precursor_molar_mass` (`float`) — Validated precursor molar mass [kg/mol].

**Returns**

- (`float`) — Finite precursor molecule number concentration [#/m³].

**Raises**

- `ValueError` — If the final float64 conversion is nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L309-L340)

## particula.dynamics.nucleation.nucleation_strategies._finite_scalar

*function*

```python
def _finite_scalar(value: object, name: str) -> float
```

Normalize a finite scalar real value.

**Parameters**

- `value` (`object`) — Candidate scalar value.
- `name` (`str`) — Parameter name used in validation errors.

**Returns**

- (`float`) — Normalized finite value.

**Raises**

- `TypeError` — If ``value`` is not an accepted scalar real value.
- `ValueError` — If conversion overflows or the value is nonfinite.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L48-L70)

## particula.dynamics.nucleation.nucleation_strategies._is_scalar_real

*function*

```python
def _is_scalar_real(value: object) -> bool
```

Return whether a value is a non-Boolean Python or NumPy real scalar.

**Parameters**

- `value` (`object`) — Candidate value to classify.

**Returns**

- (`bool`) — ``True`` when ``value`` is an accepted scalar real value.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L32-L45)

## particula.dynamics.nucleation.nucleation_strategies._nonnegative_scalar

*function*

```python
def _nonnegative_scalar(value: object, name: str) -> float
```

Normalize a finite nonnegative scalar real value.

**Parameters**

- `value` (`object`) — Candidate scalar value.
- `name` (`str`) — Parameter name used in validation errors.

**Returns**

- (`float`) — Normalized nonnegative value.

**Raises**

- `TypeError` — If ``value`` is not an accepted scalar real value.
- `ValueError` — If the value is nonfinite or negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L73-L90)

## particula.dynamics.nucleation.nucleation_strategies._positive_scalar

*function*

```python
def _positive_scalar(value: object, name: str) -> float
```

Normalize a finite positive scalar real value.

**Parameters**

- `value` (`object`) — Candidate scalar value.
- `name` (`str`) — Parameter name used in validation errors.

**Returns**

- (`float`) — Normalized positive value.

**Raises**

- `TypeError` — If ``value`` is not an accepted scalar real value.
- `ValueError` — If the value is nonfinite or not positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L93-L110)

## particula.dynamics.nucleation.nucleation_strategies._preflight_rate_inputs

*function*

```python
def _preflight_rate_inputs(validity_domain: NucleationValidityDomain, precursor_mass_concentration: object, precursor_molar_mass: object, temperature: object, saturation: object) -> tuple[float, float, float | None]
```

Validate basic rate inputs and convert precursor concentration.

Saturation is required only when its interval is configured. This preflight
intentionally precedes zero-rate and domain checks.

**Parameters**

- `validity_domain` (`NucleationValidityDomain`) — Domain that determines saturation input semantics.
- `precursor_mass_concentration` (`object`) — Candidate precursor concentration [kg/m³].
- `precursor_molar_mass` (`object`) — Candidate precursor molar mass [kg/mol].
- `temperature` (`object`) — Candidate temperature [K].
- `saturation` (`object`) — Candidate dimensionless saturation.

**Returns**

- (`float`) — Precursor number concentration [#/m³], temperature [K], and optional
- (`float`) — saturation.

**Raises**

- `TypeError` — If a required input is not an accepted scalar real value.
- `ValueError` — If a physical input is invalid, conversion is nonfinite, or
saturation is supplied outside a saturation domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L376-L427)

## particula.dynamics.nucleation.nucleation_strategies._scaled_product

*function*

```python
def _scaled_product(*factors: float, divisor: float | None = None) -> np.float64
```

Return a float64 product without intermediate range loss.

Mantissas remain near one while powers of two are accumulated separately,
so only the final ``ldexp`` can underflow or overflow.

**Parameters**

- `*factors` (`float`) (default: `()`) — Positive finite factors to multiply.
- `divisor` (`float | None`) (default: `None`) — Optional positive finite divisor.

**Returns**

- (`np.float64`) — Float64 product, including a representable subnormal final value.

**Raises**

- `FloatingPointError` — If the final result overflows or is invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L343-L373)

## particula.dynamics.nucleation.nucleation_strategies._validate_rate_domain

*function*

```python
def _validate_rate_domain(validity_domain: NucleationValidityDomain, concentration: float, temperature: float, saturation: float | None) -> bool
```

Validate nonzero-path domains and apply the lower saturation gate.

**Parameters**

- `validity_domain` (`NucleationValidityDomain`) — Inclusive concentration, temperature, and saturation constraints.
- `concentration` (`float`) — Validated precursor number concentration [#/m³].
- `temperature` (`float`) — Validated temperature [K].
- `saturation` (`float | None`) — Validated optional dimensionless saturation.

**Returns**

- (`bool`) — ``False`` only when valid saturation is below its lower interval bound.

**Raises**

- `ValueError` — If concentration, temperature, or saturation exceeds its
applicable validity domain.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L430-L471)

## particula.dynamics.nucleation.nucleation_strategies._validate_strategy_configuration

*function*

```python
def _validate_strategy_configuration(coefficient: object, validity_domain: object, injection_composition: object, formation_metadata: object, survival_factor: object, coefficient_provenance: object) -> None
```

Validate shared immutable potential-rate strategy configuration.

**Parameters**

- `coefficient` (`object`) — Nonnegative activation [s⁻¹] or kinetic [m³/s] coefficient.
- `validity_domain` (`object`) — Inclusive domain for rate evaluation.
- `injection_composition` (`object`) — Metadata for a future formed particle.
- `formation_metadata` (`object`) — Metadata for potential-rate formation size.
- `survival_factor` (`object`) — Nonnegative caller-supplied dimensionless factor.
- `coefficient_provenance` (`object`) — Optional nonempty origin description.

**Raises**

- `TypeError` — If a scalar configuration value is not a scalar real value.
- `ValueError` — If a scalar is invalid or a metadata record has wrong type.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/nucleation/nucleation_strategies.py#L270-L306)
