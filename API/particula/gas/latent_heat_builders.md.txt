# particula.gas.latent_heat_builders

*module*

Builder classes for latent heat strategy objects.

These builders mirror the vapor pressure builder pattern. They validate
required parameters, optionally convert units via
:func:`get_unit_conversion`, and construct latent heat strategies that return
J/kg values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L1-L1)

## particula.gas.latent_heat_builders.LATENT_HEAT_UNIT

*attribute* · *module attribute*

```python
LATENT_HEAT_UNIT = 'J/kg'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L27-L27)

## particula.gas.latent_heat_builders.SLOPE_UNIT

*attribute* · *module attribute*

```python
SLOPE_UNIT = 'J/(kg*K)'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L28-L28)

## particula.gas.latent_heat_builders.TEMPERATURE_UNIT

*attribute* · *module attribute*

```python
TEMPERATURE_UNIT = 'K'
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L29-L29)

## particula.gas.latent_heat_builders.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L25-L25)

## particula.gas.latent_heat_builders.ConstantLatentHeatBuilder

*class*

```python
class ConstantLatentHeatBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Build ConstantLatentHeat strategies with validated inputs.

Use :meth:`set_latent_heat_ref` to set the reference latent heat in J/kg
before calling :meth:`build`.

**Attributes**

- `latent_heat_ref` (`float | None`) — Reference latent heat in J/kg once set.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L48-L108)

### particula.gas.latent_heat_builders.ConstantLatentHeatBuilder.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L62-L62)

### particula.gas.latent_heat_builders.ConstantLatentHeatBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.latent_heat_builders.ConstantLatentHeatBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the constant latent heat builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L58-L62)

### particula.gas.latent_heat_builders.ConstantLatentHeatBuilder.build

*method*

```python
def build() -> ConstantLatentHeat
```

Validate parameters and build the ConstantLatentHeat strategy.

**Returns**

- (`ConstantLatentHeat`) — The configured ConstantLatentHeat strategy.

**Raises**

- `ValueError` — If required parameters are missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L96-L108)

### particula.gas.latent_heat_builders.ConstantLatentHeatBuilder.set_latent_heat_ref

*method*

```python
def set_latent_heat_ref(latent_heat_ref: float, latent_heat_ref_units: str | None = None) -> ConstantLatentHeatBuilder
```

Set the reference latent heat value.

**Parameters**

- `latent_heat_ref` (`float`) — Reference latent heat value.
- `latent_heat_ref_units` (`str | None`) (default: `None`) — Units for the reference latent heat. Defaults to J/kg when None.

**Returns**

- (`ConstantLatentHeatBuilder`) — The builder instance for method chaining.

**Raises**

- `ValueError` — If ``latent_heat_ref`` is not positive.
- `pint.errors.UndefinedUnitError` — If ``latent_heat_ref_units`` is
invalid and pint is available.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L64-L94)

### particula.gas.latent_heat_builders.ConstantLatentHeatBuilder.check_keys

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

### particula.gas.latent_heat_builders.ConstantLatentHeatBuilder.pre_build_check

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

### particula.gas.latent_heat_builders.ConstantLatentHeatBuilder.set_parameters

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

## particula.gas.latent_heat_builders.LinearLatentHeatBuilder

*class*

```python
class LinearLatentHeatBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Build LinearLatentHeat strategies with validated inputs.

Requires a reference latent heat, linear slope, and reference temperature.

**Attributes**

- `latent_heat_ref` (`float | None`) — Reference latent heat in J/kg once set.
- `slope` (`float | None`) — Linear slope in J/(kg*K) once set.
- `temperature_ref` (`float | None`) — Reference temperature in K once set.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L111-L235)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L126-L126)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.slope

*attribute* · *instance attribute*

```python
slope: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L127-L127)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.temperature_ref

*attribute* · *instance attribute*

```python
temperature_ref: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L128-L128)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the linear latent heat builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L122-L128)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.build

*method*

```python
def build() -> LinearLatentHeat
```

Validate parameters and build the LinearLatentHeat strategy.

**Returns**

- (`LinearLatentHeat`) — The configured LinearLatentHeat strategy.

**Raises**

- `ValueError` — If required parameters are missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L221-L235)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.set_latent_heat_ref

*method*

```python
def set_latent_heat_ref(latent_heat_ref: float, latent_heat_ref_units: str | None = None) -> LinearLatentHeatBuilder
```

Set the reference latent heat value.

**Parameters**

- `latent_heat_ref` (`float`) — Reference latent heat value.
- `latent_heat_ref_units` (`str | None`) (default: `None`) — Units for the reference latent heat. Defaults to J/kg when None.

**Returns**

- (`LinearLatentHeatBuilder`) — The builder instance for method chaining.

**Raises**

- `ValueError` — If ``latent_heat_ref`` is not positive.
- `pint.errors.UndefinedUnitError` — If ``latent_heat_ref_units`` is
invalid and pint is available.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L130-L160)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.set_slope

*method*

```python
def set_slope(slope: float, slope_units: str | None = None) -> LinearLatentHeatBuilder
```

Set the linear slope value.

**Parameters**

- `slope` (`float`) — Linear slope value.
- `slope_units` (`str | None`) (default: `None`) — Units for the slope. Defaults to J/(kg*K).

**Returns**

- (`LinearLatentHeatBuilder`) — The builder instance for method chaining.

**Raises**

- `ValueError` — If ``slope`` is not finite.
- `pint.errors.UndefinedUnitError` — If ``slope_units`` is invalid and
pint is available.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L162-L186)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.set_temperature_ref

*method*

```python
def set_temperature_ref(temperature_ref: float, temperature_ref_units: str | None = None) -> LinearLatentHeatBuilder
```

Set the reference temperature.

**Parameters**

- `temperature_ref` (`float`) — Reference temperature value.
- `temperature_ref_units` (`str | None`) (default: `None`) — Units for the temperature. Defaults to K.

**Returns**

- (`LinearLatentHeatBuilder`) — The builder instance for method chaining.

**Raises**

- `ValueError` — If ``temperature_ref`` is not positive.
- `pint.errors.UndefinedUnitError` — If ``temperature_ref_units`` is
invalid and pint is available.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L188-L219)

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.check_keys

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

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.pre_build_check

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

### particula.gas.latent_heat_builders.LinearLatentHeatBuilder.set_parameters

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

## particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder

*class*

```python
class PowerLawLatentHeatBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Build PowerLawLatentHeat strategies with validated inputs.

Requires a reference latent heat, critical temperature, and exponent.

**Attributes**

- `latent_heat_ref` (`float | None`) — Reference latent heat in J/kg once set.
- `critical_temperature` (`float | None`) — Critical temperature in K once set.
- `beta` (`float | None`) — Power-law exponent (dimensionless) once set.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L238-L366)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.beta

*attribute* · *instance attribute*

```python
beta: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L259-L259)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.critical_temperature

*attribute* · *instance attribute*

```python
critical_temperature: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L258-L258)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L257-L257)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the power-law latent heat builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L249-L259)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.build

*method*

```python
def build() -> PowerLawLatentHeat
```

Validate parameters and build the PowerLawLatentHeat strategy.

**Returns**

- (`PowerLawLatentHeat`) — The configured PowerLawLatentHeat strategy.

**Raises**

- `ValueError` — If required parameters are missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L352-L366)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.set_beta

*method*

```python
def set_beta(beta: float, beta_units: str | None = None) -> PowerLawLatentHeatBuilder
```

Set the power-law exponent.

**Parameters**

- `beta` (`float`) — Power-law exponent (dimensionless).
- `beta_units` (`str | None`) (default: `None`) — Units for the exponent. Ignored if provided.

**Returns**

- (`PowerLawLatentHeatBuilder`) — The builder instance for method chaining.

**Raises**

- `ValueError` — If ``beta`` is negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L329-L350)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.set_critical_temperature

*method*

```python
def set_critical_temperature(critical_temperature: float, critical_temperature_units: str | None = None) -> PowerLawLatentHeatBuilder
```

Set the critical temperature value.

**Parameters**

- `critical_temperature` (`float`) — Critical temperature value.
- `critical_temperature_units` (`str | None`) (default: `None`) — Units for the temperature. Defaults to K.

**Returns**

- (`PowerLawLatentHeatBuilder`) — The builder instance for method chaining.

**Raises**

- `ValueError` — If ``critical_temperature`` is not positive.
- `pint.errors.UndefinedUnitError` — If ``critical_temperature_units``
is invalid and pint is available.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L293-L327)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.set_latent_heat_ref

*method*

```python
def set_latent_heat_ref(latent_heat_ref: float, latent_heat_ref_units: str | None = None) -> PowerLawLatentHeatBuilder
```

Set the reference latent heat value.

**Parameters**

- `latent_heat_ref` (`float`) — Reference latent heat value.
- `latent_heat_ref_units` (`str | None`) (default: `None`) — Units for the reference latent heat. Defaults to J/kg when None.

**Returns**

- (`PowerLawLatentHeatBuilder`) — The builder instance for method chaining.

**Raises**

- `ValueError` — If ``latent_heat_ref`` is not positive.
- `pint.errors.UndefinedUnitError` — If ``latent_heat_ref_units`` is
invalid and pint is available.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L261-L291)

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.check_keys

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

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.pre_build_check

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

### particula.gas.latent_heat_builders.PowerLawLatentHeatBuilder.set_parameters

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

## particula.gas.latent_heat_builders._convert_value

*function*

```python
def _convert_value(value: float, unit: str, target_unit: str) -> float
```

Convert a value to the target unit using unit conversion factors.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L32-L34)

## particula.gas.latent_heat_builders._to_kelvin

*function*

```python
def _to_kelvin(temperature: float, units: str) -> float
```

Convert a temperature to Kelvin with offset-unit support.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L37-L39)

## particula.gas.latent_heat_builders._validate_temperature

*function*

```python
def _validate_temperature(name: str, temperature: float) -> None
```

Validate temperature is positive and finite after conversion.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L42-L45)
