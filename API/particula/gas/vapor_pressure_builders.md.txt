# particula.gas.vapor_pressure_builders

*module*

Builders to create vapor pressure models for gas species.

This module provides builder classes for Antoine, Clausius-Clapeyron,
constant, and WaterBuck vapor pressure strategies. Each builder follows
the same workflow:

1. Configure coefficients or parameters using dedicated methods.
2. Validate required parameters.
3. Return the corresponding vapor pressure strategy object.

> **References**
>
> - "Antoine Equation,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Antoine_equation)
> - "Clausius–Clapeyron Relation,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation)
> - "Vapor Pressure,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure)

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L1-L1)

## particula.gas.vapor_pressure_builders.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L43-L43)

## particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder

*class*

```python
class AntoineVaporPressureBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builder class for AntoineVaporPressureStrategy.

It allows setting the coefficients 'a', 'b', and 'c' separately and then
building the strategy object.
Follows the general form of the Antoine equation in

    log₁₀(P) = a − b / (T − c)

**Attributes**

- `a` — Coefficient "a" of the Antoine equation (dimensionless).
- `b` — Coefficient "b" (in Kelvin).
- `c` — Coefficient "c" (in Kelvin).

Methods:
- set_a : Set the coefficient "a" of the Antoine equation.
- set_b : Set the coefficient "b".
- set_c : Set the coefficient "c".
- build : Validate parameters and return an AntoineVaporPressureStrategy.

> **Example**
>
> ``` py title="AntoineVaporPressureBuilder"
> strategy = (
>     AntoineVaporPressureBuilder()
>     .set_a(8.07131)
>     .set_b(1730.63)
>     .set_c(233.426)
>     .build()
> )
> ```
> 
> ``` py title="AntoineVaporPressureBuilder with units"
> strategy = (
>     AntoineVaporPressureBuilder()
>     .set_a(8.07131)
>     .set_b(1730.63, "K")
>     .set_c(233.426, "K")
>     .build()
> )
> ```

> **References**
>
> - Equation: log10(P_mmHG) = a - b / (Temperature_K - c)
>   (Reference: https://en.wikipedia.org/wiki/Antoine_equation)
> - "Vapor Pressure,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure)
> - "Atmospheric Pressure Unit Conversions,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Pascal_(unit))

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L46-L149)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.a

*attribute* · *instance attribute*

```python
a = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L105-L105)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.b

*attribute* · *instance attribute*

```python
b = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L106-L106)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.c

*attribute* · *instance attribute*

```python
c = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L107-L107)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Antoine vapor pressure builder.

Sets up the builder with required parameters (a, b, c) for
creating an AntoineVaporPressureStrategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L96-L107)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.build

*method*

```python
def build() -> AntoineVaporPressureStrategy
```

Validate and return an AntoineVaporPressureStrategy using the set
coefficients.

**Returns**

- (`AntoineVaporPressureStrategy`) — Configured with coefficients a, b, and c.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L141-L149)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.set_a

*method*

```python
def set_a(a: float, a_units: Optional[str] = None) -> AntoineVaporPressureBuilder
```

Set the coefficient 'a' of the Antoine equation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L109-L119)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.set_b

*method*

```python
def set_b(b: float, b_units: str = 'K') -> AntoineVaporPressureBuilder
```

Set the coefficient 'b' of the Antoine equation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L121-L129)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.set_c

*method*

```python
def set_c(c: float, c_units: str = 'K') -> AntoineVaporPressureBuilder
```

Set the coefficient 'c' of the Antoine equation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L131-L139)

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.check_keys

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

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.pre_build_check

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

### particula.gas.vapor_pressure_builders.AntoineVaporPressureBuilder.set_parameters

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

## particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder

*class*

```python
class ClausiusClapeyronVaporPressureBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builder class for ClausiusClapeyronStrategy. This class facilitates
setting the latent heat of vaporization, initial temperature, and initial
pressure with unit handling and then builds the strategy object.

The Clausius–Clapeyron relation can be approximated as:

- dP / dT = (L / (R × T²))

Methods:
- set_latent_heat : Set latent heat in J/mol (or convertible units).
- set_temperature_initial : Set initial temperature in K
    (or convertible units).
- set_pressure_initial : Set initial pressure in Pa
    (or convertible units).
- build : Validate parameters and return a ClausiusClapeyronStrategy.

> **Example**
>
> ``` py title="ClausiusClapeyronVaporPressureBuilder"
> strategy = (
>     ClausiusClapeyronVaporPressureBuilder()
>     .set_latent_heat(2260)
>     .set_temperature_initial(373.15)
>     .set_pressure_initial(101325)
>     .build()
> )
> ```
> 
> ``` py title="ClausiusClapeyronVaporPressureBuilder with units"
> strategy = (
>     ClausiusClapeyronVaporPressureBuilder()
>     .set_latent_heat(2260, "J/mol")
>     .set_temperature_initial(373.15, "K")
>     .set_pressure_initial(101325, "Pa")
>     .build()
> )
> ```

> **References**
>
> - Equation: dP/dT = L / (R * T^2)
>   https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L152-L264)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.latent_heat

*attribute* · *instance attribute*

```python
latent_heat = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L210-L210)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.pressure_initial

*attribute* · *instance attribute*

```python
pressure_initial = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L212-L212)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.temperature_initial

*attribute* · *instance attribute*

```python
temperature_initial = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L211-L211)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Clausius-Clapeyron vapor pressure builder.

Sets up the builder with required parameters for creating a
ClausiusClapeyronStrategy, including latent heat, initial
temperature, and initial pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L197-L212)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.build

*method*

```python
def build() -> ClausiusClapeyronStrategy
```

Validate parameters and return a ClausiusClapeyronStrategy object.

**Returns**

- (`ClausiusClapeyronStrategy`) — Configured with latent heat, initial Temperature, and Pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L253-L264)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.set_latent_heat

*method*

```python
def set_latent_heat(latent_heat: float, latent_heat_units: str) -> ClausiusClapeyronVaporPressureBuilder
```

Set the latent heat of vaporization: Default units J/mol.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L214-L225)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.set_pressure_initial

*method*

```python
def set_pressure_initial(pressure_initial: float, pressure_initial_units: str) -> ClausiusClapeyronVaporPressureBuilder
```

Set the initial pressure. Default units: Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L240-L251)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.set_temperature_initial

*method*

```python
def set_temperature_initial(temperature_initial: float, temperature_initial_units: str) -> ClausiusClapeyronVaporPressureBuilder
```

Set the initial temperature. Default units: K.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L227-L238)

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.check_keys

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

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.pre_build_check

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

### particula.gas.vapor_pressure_builders.ClausiusClapeyronVaporPressureBuilder.set_parameters

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

## particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder

*class*

```python
class ConstantVaporPressureBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builder class for ConstantVaporPressureStrategy. This class facilitates
setting the constant vapor pressure and then building the strategy object.

**Attributes**

- `vapor_pressure` — The vapor pressure in Pa (scalar/float).

- set_vapor_pressure: Set the constant vapor pressure in Pa
  (or convertible units).
- build : Validate parameters and return a ConstantVaporPressureStrategy.

> **Example**
>
> ``` py title="ConstantVaporPressureBuilder"
> strategy = (
>     ConstantVaporPressureBuilder()
>     .set_vapor_pressure(101325)
>     .build()
> )
> ```
> 
> ``` py title="ConstantVaporPressureBuilder with units"
> strategy = (
>     ConstantVaporPressureBuilder()
>     .set_vapor_pressure(1, "atm")
>     .build()
> )
> ```

> **References**
>
> - Equation: P = vapor_pressure
>   https://en.wikipedia.org/wiki/Vapor_pressure

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L267-L330)

### particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder.vapor_pressure

*attribute* · *instance attribute*

```python
vapor_pressure = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L308-L308)

### particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Constant vapor pressure builder.

Sets up the builder with the required vapor_pressure parameter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L301-L308)

### particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder.build

*method*

```python
def build() -> ConstantVaporPressureStrategy
```

Validate parameters and return ConstantVaporPressureStrategy.

**Returns**

- (`ConstantVaporPressureStrategy`) — Configured with vapor_pressure in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L323-L330)

### particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder.set_vapor_pressure

*method*

```python
def set_vapor_pressure(vapor_pressure: float, vapor_pressure_units: str) -> ConstantVaporPressureBuilder
```

Set the constant vapor pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L310-L321)

### particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder.check_keys

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

### particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder.pre_build_check

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

### particula.gas.vapor_pressure_builders.ConstantVaporPressureBuilder.set_parameters

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

## particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder

*class*

```python
class SaturationConcentrationVaporPressureBuilder(BuilderABC, BuilderMolarMassMixin, BuilderTemperatureMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.builder_mixin.BuilderTemperatureMixin`

Builder class for ConstantVaporPressureStrategy.

It allows for setting the vapor pressure using the saturation
concentration, molar mass, and temperature. The saturation concentration is
commonly called C^sat (or C* when in a mixture) in aerosol sciences.

- set_saturation_concentration : Set the saturation concentration in
  kg/m³ (or convertible units).
- set_molar_mass : Set the molar mass in kg/mol (or convertible units).
- set_temperature : Set the temperature in K (or convertible units).
- build : Validate parameters and return a ConstantVaporPressureStrategy.

> **Example**
>
> ```py title="SaturationConcentrationVaporPressureBuilder"
> import particula as par
> strategy = (
>     par.gas.SaturationConcentrationVaporPressureBuilder()
>     .set_saturation_concentration(10e-6, "kg/m^3")
>     .set_molar_mass(0.200, "kg/mol")
>     .set_temperature(298.15, "K")
>     .build()
> )
> ```

References:
- Donahue, N. M., Robinson, A. L., Stanier, C. O., & Pandis, S. N. (2006).
  Coupled Partitioning, Dilution, and Chemical Aging of Semivolatile
  Organics. Environmental Science & Technology, 40(8), 2635–2643.
  [DOI](https://doi.org/10.1021/es052297c)

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L333-L439)

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L386-L386)

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.saturation_concentration

*attribute* · *instance attribute*

```python
saturation_concentration = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L385-L385)

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.temperature

*attribute* · *instance attribute*

```python
temperature = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L387-L387)

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Saturation Concentration vapor pressure builder.

Sets up the builder with required parameters for creating a
ConstantVaporPressureStrategy from saturation concentration,
molar mass, and temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L370-L387)

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.build

*method*

```python
def build() -> ConstantVaporPressureStrategy
```

Validate all parameters and construct a
ConstantVaporPressureStrategy whose value is obtained from the
ideal-gas relationship between concentration and partial pressure.

**Returns**

- (`ConstantVaporPressureStrategy`) — ConstantVaporPressureStrategy containing the calculated vapor
- (`ConstantVaporPressureStrategy`) — pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L423-L439)

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.set_saturation_concentration

*method*

```python
def set_saturation_concentration(saturation_concentration: float, saturation_concentration_units: str) -> SaturationConcentrationVaporPressureBuilder
```

Set the saturation concentration (C*, C^sat).

**Parameters**

- `saturation_concentration` (`float`) — Value of the saturation concentration.
- `saturation_concentration_units` (`str`) — Any units convertible to ``"kg/m^3"`` via ``get_unit_conversion`` are accepted.

**Returns**

- (`SaturationConcentrationVaporPressureBuilder`) — - The builder itself for fluent chaining.

> **Example**
>
> ```py title="SaturationConcentrationVaporPressureBuilder"
> import particula as par
> strategy = (
>     par.gas.SaturationConcentrationVaporPressureBuilder()
>     .set_saturation_concentration(10e-6, "kg/m^3")
> )
> ```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L389-L421)

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.check_keys

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

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.pre_build_check

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

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.set_parameters

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

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.set_molar_mass

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

### particula.gas.vapor_pressure_builders.SaturationConcentrationVaporPressureBuilder.set_temperature

*method*

```python
def set_temperature(temperature: float, temperature_units: str = 'K') -> Self
```

Inherited from: `particula.builder_mixin.BuilderTemperatureMixin`

Set the temperature in Kelvin.

**Parameters**

- `temperature` (`float`) — Finite temperature value.
- `temperature_units` (`str`) (default: `'K'`) — Units for ``temperature``. Defaults to ``"K"``. Accepts ``"degC"``, ``"degF"``, ``"degR"``, or ``"K"``.

**Returns**

- `self` (`BuilderTemperatureMixin`) — Self for method chaining.

**Raises**

- `ValueError` — If the converted temperature is below absolute zero.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L389-L414)

## particula.gas.vapor_pressure_builders.TableVaporPressureBuilder

*class*

```python
class TableVaporPressureBuilder(BuilderABC, BuilderTemperatureTableMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderTemperatureTableMixin`

Builder for TableVaporPressureStrategy.

Allows setting lookup tables of vapor pressure and temperature, handling
optional unit conversion before constructing the strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L472-L517)

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.vapor_pressure_table

*attribute* · *instance attribute*

```python
vapor_pressure_table: NDArray[np.float64] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L492-L492)

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.temperature_table

*attribute* · *instance attribute*

```python
temperature_table: Optional[NDArray[np.float64]] = None
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L426-L426)

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the Table vapor pressure builder.

Sets up the builder with required parameters for creating a
TableVaporPressureStrategy from lookup tables of vapor pressure
and temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L482-L492)

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.build

*method*

```python
def build() -> TableVaporPressureStrategy
```

Validate and construct a TableVaporPressureStrategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L510-L517)

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.set_vapor_pressure_table

*method*

```python
def set_vapor_pressure_table(vapor_pressure_table: NDArray[np.float64], vapor_pressure_table_units: str = 'Pa') -> TableVaporPressureBuilder
```

Set the vapor pressure lookup table.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L494-L508)

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.check_keys

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

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.pre_build_check

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

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.set_parameters

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

### particula.gas.vapor_pressure_builders.TableVaporPressureBuilder.set_temperature_table

*method*

```python
def set_temperature_table(temperature_table: NDArray[np.float64], temperature_table_units: str = 'K')
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

Assign a table of temperatures in Kelvin.

**Parameters**

- `temperature_table` (`NDArray[np.float64]`) — Array of temperature values.
- `temperature_table_units` (`str`) (default: `'K'`) — Units for the values. Defaults to ``"K"``.

**Returns**

- `self` (`BuilderTemperatureTableMixin`) — Self for method chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L428-L453)

## particula.gas.vapor_pressure_builders.WaterBuckVaporPressureBuilder

*class*

```python
class WaterBuckVaporPressureBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Builder class for WaterBuckStrategy.

This class facilitates the building of the WaterBuckStrategy object.
Which as of now has no additional parameters to set but could be
extended in the future (e.g., ice-only calculations).

> **Example**
>
> ```py title="WaterBuckVaporPressureBuilder"
> import particula as par
> strategy = par.gas.WaterBuckVaporPressureBuilder().build()
> ```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L442-L469)

### particula.gas.vapor_pressure_builders.WaterBuckVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.vapor_pressure_builders.WaterBuckVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Water Buck vapor pressure builder.

Sets up the builder for creating a WaterBuckStrategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L456-L461)

### particula.gas.vapor_pressure_builders.WaterBuckVaporPressureBuilder.build

*method*

```python
def build() -> WaterBuckStrategy
```

Build and return a WaterBuckStrategy object.

**Returns**

- (`WaterBuckStrategy`) — Configured for water-specific Buck vapor pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L463-L469)

### particula.gas.vapor_pressure_builders.WaterBuckVaporPressureBuilder.check_keys

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

### particula.gas.vapor_pressure_builders.WaterBuckVaporPressureBuilder.pre_build_check

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

### particula.gas.vapor_pressure_builders.WaterBuckVaporPressureBuilder.set_parameters

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
