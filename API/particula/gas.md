# particula.gas

*module*

Expose the public gas-phase API for particula.

This module provides classes and functions for gas-phase modeling,
including native gas and environment containers, gas data construction,
and vapor-pressure and latent-heat strategies.

> **Example**
>
> >>> from particula.gas import EnvironmentData, GasDataBuilder
> >>> environment = EnvironmentData(
> ...     temperature=[298.15],
> ...     pressure=[101325.0],
> ...     saturation_ratio=[[0.5]],
> ... )
> >>> gas = (
> ...     GasDataBuilder()
> ...     .set_names(["Water"])
> ...     .set_molar_mass([18], units="g/mol")
> ...     .set_concentration([1e-6], units="kg/m^3")
> ...     .set_partitioning([True])
> ...     .build()
> ... )
> >>> environment.n_boxes
> 1

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/__init__.py#L1-L1)

## particula.gas.AntoineVaporPressureBuilder

*class*

```python
class AntoineVaporPressureBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.gas.vapor_pressure_builders`

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

### particula.gas.AntoineVaporPressureBuilder.a

*attribute* · *instance attribute*

```python
a = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L105-L105)

### particula.gas.AntoineVaporPressureBuilder.b

*attribute* · *instance attribute*

```python
b = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L106-L106)

### particula.gas.AntoineVaporPressureBuilder.c

*attribute* · *instance attribute*

```python
c = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L107-L107)

### particula.gas.AntoineVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.AntoineVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Antoine vapor pressure builder.

Sets up the builder with required parameters (a, b, c) for
creating an AntoineVaporPressureStrategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L96-L107)

### particula.gas.AntoineVaporPressureBuilder.build

*method*

```python
def build() -> AntoineVaporPressureStrategy
```

Validate and return an AntoineVaporPressureStrategy using the set
coefficients.

**Returns**

- (`AntoineVaporPressureStrategy`) — Configured with coefficients a, b, and c.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L141-L149)

### particula.gas.AntoineVaporPressureBuilder.set_a

*method*

```python
def set_a(a: float, a_units: Optional[str] = None) -> AntoineVaporPressureBuilder
```

Set the coefficient 'a' of the Antoine equation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L109-L119)

### particula.gas.AntoineVaporPressureBuilder.set_b

*method*

```python
def set_b(b: float, b_units: str = 'K') -> AntoineVaporPressureBuilder
```

Set the coefficient 'b' of the Antoine equation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L121-L129)

### particula.gas.AntoineVaporPressureBuilder.set_c

*method*

```python
def set_c(c: float, c_units: str = 'K') -> AntoineVaporPressureBuilder
```

Set the coefficient 'c' of the Antoine equation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L131-L139)

### particula.gas.AntoineVaporPressureBuilder.check_keys

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

### particula.gas.AntoineVaporPressureBuilder.pre_build_check

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

### particula.gas.AntoineVaporPressureBuilder.set_parameters

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

## particula.gas.AntoineVaporPressureStrategy

*class*

```python
class AntoineVaporPressureStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Re-exported from: `particula.gas.vapor_pressure_strategies`

Vapor pressure strategy using the Antoine equation.

This class calculates vapor pressure by applying the Antoine equation,
which relates temperature in Kelvin to the logarithm of vapor pressure.

**Attributes**

- `a` — Coefficient "a" in the Antoine equation.
- `b` — Coefficient "b" in the Antoine equation.
- `c` — Coefficient "c" in the Antoine equation.

Methods:
- partial_pressure: Compute partial pressure from concentration.
- concentration: Compute concentration from partial pressure.
- saturation_ratio: Compute ratio of partial pressure to saturation
  pressure.
- saturation_concentration: Compute concentration at saturation pressure.
- pure_vapor_pressure: Computes vapor pressure from the Antoine equation.

**Examples**

```py title="Antoine Vapor Pressure Example"
import particula as par
strategy = par.gas.AntoineVaporPressureStrategy(
    a=8.07131, b=1730.63, c=233.426
)
vp = strategy.pure_vapor_pressure(temperature=373.15)
# Returns the vapor pressure in Pascals
```

> **References**
>
> - "Antoine Equation,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Antoine_equation).
> - Kelvin-based adaptation:
>   https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L246-L327)

### particula.gas.AntoineVaporPressureStrategy.MMHG_TO_PA

*attribute* · *class attribute* · *instance attribute*

```python
MMHG_TO_PA = 133.322
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L282-L282)

### particula.gas.AntoineVaporPressureStrategy.a

*attribute* · *instance attribute*

```python
a = a
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L297-L297)

### particula.gas.AntoineVaporPressureStrategy.b

*attribute* · *instance attribute*

```python
b = b
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L298-L298)

### particula.gas.AntoineVaporPressureStrategy.c

*attribute* · *instance attribute*

```python
c = c
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L299-L299)

### particula.gas.AntoineVaporPressureStrategy.__init__

*method*

```python
def __init__(a: Union[float, NDArray[np.float64]] = 0.0, b: Union[float, NDArray[np.float64]] = 0.0, c: Union[float, NDArray[np.float64]] = 0.0)
```

Initialize the Antoine Vapor Pressure strategy.

**Parameters**

- `a` (`Union[float, NDArray[np.float64]]`) (default: `0.0`) — Coefficient 'a' in the Antoine equation.
- `b` (`Union[float, NDArray[np.float64]]`) (default: `0.0`) — Coefficient 'b' in the Antoine equation (K).
- `c` (`Union[float, NDArray[np.float64]]`) (default: `0.0`) — Coefficient 'c' in the Antoine equation (K).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L284-L299)

### particula.gas.AntoineVaporPressureStrategy.pure_vapor_pressure

*method*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate vapor pressure using the Antoine equation.

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure in Pascals.

**Examples**

``` py title="Antoine Vapor Pressure Calculation"
vapor_pressure = strategy.pure_vapor_pressure(
    temperature=300
)
```

> **References**
>
> - Equation: log10(P) = a - b / (T - c)
> - https://en.wikipedia.org/wiki/Antoine_equation (but in Kelvin)
> - Kelvin form:
>     https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L301-L327)

### particula.gas.AntoineVaporPressureStrategy.concentration

*method*

```python
def concentration(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the concentration of the gas at a given pressure and
temperature.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Pressure in Pascals.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The concentration of the gas in kg/m^3.

**Examples**

``` py title="Concentration Calculation"
concentration = strategy.concentration(
    partial_pressure=101325,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L87-L115)

### particula.gas.AntoineVaporPressureStrategy.partial_pressure

*method*

```python
def partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate partial pressure from concentration, molar mass, T.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals.

**Examples**

``` py title="Partial Pressure Calculation"
partial_pressure = strategy.partial_pressure(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L60-L85)

### particula.gas.AntoineVaporPressureStrategy.saturation_concentration

*method*

```python
def saturation_concentration(molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation concentration of the gas at a given
temperature.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation concentration of the gas in kg/m^3.

**Examples**

``` py title="Saturation Concentration Calculation"
saturation_concentration = strategy.saturation_concentration(
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L147-L172)

### particula.gas.AntoineVaporPressureStrategy.saturation_ratio

*method*

```python
def saturation_ratio(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation ratio of the gas from its concentration and
molar mass at a given temperature.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation ratio of the gas.

**Examples**

``` py title="Saturation Ratio Calculation"
saturation_ratio = strategy.saturation_ratio(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L117-L145)

## particula.gas.ArblasterLiquidVaporPressureStrategy

*class*

```python
class ArblasterLiquidVaporPressureStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Re-exported from: `particula.gas.vapor_pressure_strategies`

Vapor pressure strategy using a 5-term logarithmic polynomial.

This strategy applies the Honig–Kramer/Arblaster correlation for
liquid metals. The relationship is expressed as:

- ln(P[bar]) = A + B ln(T) + C ∕ T + D T + E T²
    - P is the vapor pressure in bar,
    - T is the temperature in kelvin,
    - A, B, C, D and E are empirical coefficients.

Methods:
- pure_vapor_pressure: Compute the vapor pressure using the coefficients.

**Examples**

```py title="Instantiate the Arblaster strategy"
coeffs = (7.5, 1.0, -2000.0, 0.002, -1e-6)
strategy = ArblasterLiquidVaporPressureStrategy(coeffs)
vp = strategy.pure_vapor_pressure(3000.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L543-L597)

### particula.gas.ArblasterLiquidVaporPressureStrategy.coefficients

*attribute* · *instance attribute*

```python
coefficients = coefficients
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L571-L571)

### particula.gas.ArblasterLiquidVaporPressureStrategy.__init__

*method*

```python
def __init__(coefficients: tuple[float, float, float, float, float])
```

Initialize with the 5-term polynomial coefficients.

**Parameters**

- `coefficients` (`tuple[float, float, float, float, float]`) — Tuple ``(A, B, C, D, E)`` for the correlation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L565-L571)

### particula.gas.ArblasterLiquidVaporPressureStrategy.pure_vapor_pressure

*method*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate vapor pressure from the 5-term polynomial.

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure in Pascals.

**Examples**

``` py title="Vapor pressure from polynomial"
vp = strategy.pure_vapor_pressure(3000.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L573-L597)

### particula.gas.ArblasterLiquidVaporPressureStrategy.concentration

*method*

```python
def concentration(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the concentration of the gas at a given pressure and
temperature.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Pressure in Pascals.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The concentration of the gas in kg/m^3.

**Examples**

``` py title="Concentration Calculation"
concentration = strategy.concentration(
    partial_pressure=101325,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L87-L115)

### particula.gas.ArblasterLiquidVaporPressureStrategy.partial_pressure

*method*

```python
def partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate partial pressure from concentration, molar mass, T.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals.

**Examples**

``` py title="Partial Pressure Calculation"
partial_pressure = strategy.partial_pressure(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L60-L85)

### particula.gas.ArblasterLiquidVaporPressureStrategy.saturation_concentration

*method*

```python
def saturation_concentration(molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation concentration of the gas at a given
temperature.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation concentration of the gas in kg/m^3.

**Examples**

``` py title="Saturation Concentration Calculation"
saturation_concentration = strategy.saturation_concentration(
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L147-L172)

### particula.gas.ArblasterLiquidVaporPressureStrategy.saturation_ratio

*method*

```python
def saturation_ratio(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation ratio of the gas from its concentration and
molar mass at a given temperature.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation ratio of the gas.

**Examples**

``` py title="Saturation Ratio Calculation"
saturation_ratio = strategy.saturation_ratio(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L117-L145)

## particula.gas.ClausiusClapeyronStrategy

*class*

```python
class ClausiusClapeyronStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Re-exported from: `particula.gas.vapor_pressure_strategies`

Vapor pressure strategy using the Clausius-Clapeyron equation.

This class calculates vapor pressure by applying the Clausius-Clapeyron
relation, which relates how the vapor pressure of a substance changes
with temperature, given latent heat data and a reference point.

**Attributes**

- `latent_heat` — Latent heat of vaporization (J/mol).
- `temperature_initial` — Reference temperature (K).
- `pressure_initial` — Reference pressure (Pa).

Methods:
- partial_pressure: Compute partial pressure from concentration.
- concentration: Compute concentration from partial pressure.
- saturation_ratio: Compute ratio of partial pressure to saturation
  pressure.
- saturation_concentration: Compute concentration at saturation pressure.
- pure_vapor_pressure: Computes vapor pressure via Clausius-Clapeyron
  relation.

**Examples**

```py title="Clausius-Clapeyron Example"
strategy = ClausiusClapeyronStrategy(
    latent_heat=4.07e4,
    temperature_initial=298.15,
    pressure_initial=3167.0
)
vp = strategy.pure_vapor_pressure(temperature=310)
```

> **References**
>
> - "Clausius–Clapeyron relation,"
>   [Wikipedia](https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L330-L410)

### particula.gas.ClausiusClapeyronStrategy.latent_heat

*attribute* · *instance attribute*

```python
latent_heat = latent_heat
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L380-L380)

### particula.gas.ClausiusClapeyronStrategy.pressure_initial

*attribute* · *instance attribute*

```python
pressure_initial = pressure_initial
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L382-L382)

### particula.gas.ClausiusClapeyronStrategy.temperature_initial

*attribute* · *instance attribute*

```python
temperature_initial = temperature_initial
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L381-L381)

### particula.gas.ClausiusClapeyronStrategy.__init__

*method*

```python
def __init__(latent_heat: Union[float, NDArray[np.float64]], temperature_initial: Union[float, NDArray[np.float64]], pressure_initial: Union[float, NDArray[np.float64]])
```

Initializes the Clausius-Clapeyron strategy with the specific latent
heat of vaporization and the specific gas constant of the substance.

**Parameters**

- `latent_heat` (`Union[float, NDArray[np.float64]]`) — Latent heat of vaporization in J/mol.
- `temperature_initial` (`Union[float, NDArray[np.float64]]`) — Initial temperature in Kelvin.
- `pressure_initial` (`Union[float, NDArray[np.float64]]`) — Initial vapor pressure in Pascals.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L366-L382)

### particula.gas.ClausiusClapeyronStrategy.pure_vapor_pressure

*method*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate vapor pressure using Clausius-Clapeyron equation.

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Final temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure in Pascals.

**Examples**

``` py title="Clausius-Clapeyron Vapor Pressure Calculation"
vapor_pressure = strategy.pure_vapor_pressure(
    temperature=300
)
```

> **References**
>
> - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L384-L410)

### particula.gas.ClausiusClapeyronStrategy.concentration

*method*

```python
def concentration(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the concentration of the gas at a given pressure and
temperature.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Pressure in Pascals.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The concentration of the gas in kg/m^3.

**Examples**

``` py title="Concentration Calculation"
concentration = strategy.concentration(
    partial_pressure=101325,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L87-L115)

### particula.gas.ClausiusClapeyronStrategy.partial_pressure

*method*

```python
def partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate partial pressure from concentration, molar mass, T.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals.

**Examples**

``` py title="Partial Pressure Calculation"
partial_pressure = strategy.partial_pressure(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L60-L85)

### particula.gas.ClausiusClapeyronStrategy.saturation_concentration

*method*

```python
def saturation_concentration(molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation concentration of the gas at a given
temperature.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation concentration of the gas in kg/m^3.

**Examples**

``` py title="Saturation Concentration Calculation"
saturation_concentration = strategy.saturation_concentration(
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L147-L172)

### particula.gas.ClausiusClapeyronStrategy.saturation_ratio

*method*

```python
def saturation_ratio(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation ratio of the gas from its concentration and
molar mass at a given temperature.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation ratio of the gas.

**Examples**

``` py title="Saturation Ratio Calculation"
saturation_ratio = strategy.saturation_ratio(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L117-L145)

## particula.gas.ClausiusClapeyronVaporPressureBuilder

*class*

```python
class ClausiusClapeyronVaporPressureBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.gas.vapor_pressure_builders`

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

### particula.gas.ClausiusClapeyronVaporPressureBuilder.latent_heat

*attribute* · *instance attribute*

```python
latent_heat = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L210-L210)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.pressure_initial

*attribute* · *instance attribute*

```python
pressure_initial = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L212-L212)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.temperature_initial

*attribute* · *instance attribute*

```python
temperature_initial = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L211-L211)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Clausius-Clapeyron vapor pressure builder.

Sets up the builder with required parameters for creating a
ClausiusClapeyronStrategy, including latent heat, initial
temperature, and initial pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L197-L212)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.build

*method*

```python
def build() -> ClausiusClapeyronStrategy
```

Validate parameters and return a ClausiusClapeyronStrategy object.

**Returns**

- (`ClausiusClapeyronStrategy`) — Configured with latent heat, initial Temperature, and Pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L253-L264)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.set_latent_heat

*method*

```python
def set_latent_heat(latent_heat: float, latent_heat_units: str) -> ClausiusClapeyronVaporPressureBuilder
```

Set the latent heat of vaporization: Default units J/mol.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L214-L225)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.set_pressure_initial

*method*

```python
def set_pressure_initial(pressure_initial: float, pressure_initial_units: str) -> ClausiusClapeyronVaporPressureBuilder
```

Set the initial pressure. Default units: Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L240-L251)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.set_temperature_initial

*method*

```python
def set_temperature_initial(temperature_initial: float, temperature_initial_units: str) -> ClausiusClapeyronVaporPressureBuilder
```

Set the initial temperature. Default units: K.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L227-L238)

### particula.gas.ClausiusClapeyronVaporPressureBuilder.check_keys

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

### particula.gas.ClausiusClapeyronVaporPressureBuilder.pre_build_check

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

### particula.gas.ClausiusClapeyronVaporPressureBuilder.set_parameters

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

## particula.gas.ConstantLatentHeat

*class*

```python
class ConstantLatentHeat(LatentHeatStrategy)
```

Bases: `particula.gas.latent_heat_strategies.LatentHeatStrategy`

Re-exported from: `particula.gas.latent_heat_strategies`

Latent heat strategy that returns a constant value.

**Attributes**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg returned by the strategy.

**Examples**

```py title="ConstantLatentHeat usage example"
import numpy as np
from particula.gas.latent_heat_strategies import ConstantLatentHeat

strategy = ConstantLatentHeat(latent_heat_ref=2.26e6)
print(strategy.latent_heat(298.15))
print(strategy.latent_heat(np.array([280.0, 300.0])))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L37-L84)

### particula.gas.ConstantLatentHeat.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float = latent_heat_ref
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L62-L62)

### particula.gas.ConstantLatentHeat.__init__

*method*

```python
def __init__(latent_heat_ref: float) -> None
```

Initialize the constant latent heat strategy.

**Parameters**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L55-L62)

### particula.gas.ConstantLatentHeat.latent_heat

*method*

```python
def latent_heat(temperature: float | NDArray[np.float64]) -> float | NDArray[np.float64]
```

Return a constant latent heat value.

**Parameters**

- `temperature` (`float | NDArray[np.float64]`) — Temperature in Kelvin. Accepted for interface consistency.

**Returns**

- (`float | NDArray[np.float64]`) — Latent heat in J/kg. Returns a scalar for scalar input or an array
- (`float | NDArray[np.float64]`) — matching the input shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L64-L84)

## particula.gas.ConstantLatentHeatBuilder

*class*

```python
class ConstantLatentHeatBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.gas.latent_heat_builders`

Build ConstantLatentHeat strategies with validated inputs.

Use :meth:`set_latent_heat_ref` to set the reference latent heat in J/kg
before calling :meth:`build`.

**Attributes**

- `latent_heat_ref` (`float | None`) — Reference latent heat in J/kg once set.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L48-L108)

### particula.gas.ConstantLatentHeatBuilder.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L62-L62)

### particula.gas.ConstantLatentHeatBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.ConstantLatentHeatBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the constant latent heat builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L58-L62)

### particula.gas.ConstantLatentHeatBuilder.build

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

### particula.gas.ConstantLatentHeatBuilder.set_latent_heat_ref

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

### particula.gas.ConstantLatentHeatBuilder.check_keys

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

### particula.gas.ConstantLatentHeatBuilder.pre_build_check

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

### particula.gas.ConstantLatentHeatBuilder.set_parameters

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

## particula.gas.ConstantVaporPressureBuilder

*class*

```python
class ConstantVaporPressureBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.gas.vapor_pressure_builders`

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

### particula.gas.ConstantVaporPressureBuilder.vapor_pressure

*attribute* · *instance attribute*

```python
vapor_pressure = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L308-L308)

### particula.gas.ConstantVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.ConstantVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Constant vapor pressure builder.

Sets up the builder with the required vapor_pressure parameter.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L301-L308)

### particula.gas.ConstantVaporPressureBuilder.build

*method*

```python
def build() -> ConstantVaporPressureStrategy
```

Validate parameters and return ConstantVaporPressureStrategy.

**Returns**

- (`ConstantVaporPressureStrategy`) — Configured with vapor_pressure in Pa.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L323-L330)

### particula.gas.ConstantVaporPressureBuilder.set_vapor_pressure

*method*

```python
def set_vapor_pressure(vapor_pressure: float, vapor_pressure_units: str) -> ConstantVaporPressureBuilder
```

Set the constant vapor pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L310-L321)

### particula.gas.ConstantVaporPressureBuilder.check_keys

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

### particula.gas.ConstantVaporPressureBuilder.pre_build_check

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

### particula.gas.ConstantVaporPressureBuilder.set_parameters

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

## particula.gas.ConstantVaporPressureStrategy

*class*

```python
class ConstantVaporPressureStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Re-exported from: `particula.gas.vapor_pressure_strategies`

Vapor pressure strategy with a constant value.

This class returns a single, unchanging vapor pressure value regardless of
the temperature. It is useful for scenarios that require a simplified
model.

**Attributes**

- `vapor_pressure` — The constant vapor pressure in Pascals.

Methods:
- partial_pressure: Compute partial pressure from concentration.
- concentration: Compute concentration from partial pressure.
- saturation_ratio: Compute ratio of partial pressure to saturation
  pressure.
- saturation_concentration: Compute concentration at saturation pressure.
- pure_vapor_pressure: Returns the constant vapor pressure.

**Examples**

```py title="Constant Vapor Pressure Example"
import particula as par
strategy = par.gas.ConstantVaporPressureStrategy(101325.0)
vp = strategy.pure_vapor_pressure(temperature=300)
# vp is 101325.0
```

> **References**
>
> - None

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L186-L243)

### particula.gas.ConstantVaporPressureStrategy.vapor_pressure

*attribute* · *instance attribute*

```python
vapor_pressure = vapor_pressure
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L222-L222)

### particula.gas.ConstantVaporPressureStrategy.__init__

*method*

```python
def __init__(vapor_pressure: Union[float, NDArray[np.float64]])
```

Initialize the Constant Vapor Pressure strategy.

**Parameters**

- `vapor_pressure` (`Union[float, NDArray[np.float64]]`) — The constant vapor pressure value in Pascals.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L216-L222)

### particula.gas.ConstantVaporPressureStrategy.pure_vapor_pressure

*method*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Return the constant vapor pressure value.

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Not used.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The constant vapor pressure value in Pascals.

**Examples**

``` py title="Constant Vapor Pressure Calculation"
vapor_pressure = strategy.pure_vapor_pressure(
    temperature=300
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L224-L243)

### particula.gas.ConstantVaporPressureStrategy.concentration

*method*

```python
def concentration(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the concentration of the gas at a given pressure and
temperature.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Pressure in Pascals.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The concentration of the gas in kg/m^3.

**Examples**

``` py title="Concentration Calculation"
concentration = strategy.concentration(
    partial_pressure=101325,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L87-L115)

### particula.gas.ConstantVaporPressureStrategy.partial_pressure

*method*

```python
def partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate partial pressure from concentration, molar mass, T.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals.

**Examples**

``` py title="Partial Pressure Calculation"
partial_pressure = strategy.partial_pressure(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L60-L85)

### particula.gas.ConstantVaporPressureStrategy.saturation_concentration

*method*

```python
def saturation_concentration(molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation concentration of the gas at a given
temperature.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation concentration of the gas in kg/m^3.

**Examples**

``` py title="Saturation Concentration Calculation"
saturation_concentration = strategy.saturation_concentration(
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L147-L172)

### particula.gas.ConstantVaporPressureStrategy.saturation_ratio

*method*

```python
def saturation_ratio(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation ratio of the gas from its concentration and
molar mass at a given temperature.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation ratio of the gas.

**Examples**

``` py title="Saturation Ratio Calculation"
saturation_ratio = strategy.saturation_ratio(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L117-L145)

## particula.gas.EnvironmentData

*class* · *dataclass*

```python
class EnvironmentData
```

Re-exported from: `particula.gas.environment_data`

Mutable thermodynamic environment data for one or more boxes.

This class stores box-level temperature and pressure together with a
per-box, per-species saturation-ratio array. The temperature array is
the source of truth for the box axis, so derived properties such as
``n_boxes`` follow ``temperature.shape[0]``.

**Attributes**

- `temperature` (`NDArray[np.float64]`) — Box temperatures in kelvin. Shape: ``(n_boxes,)``.
- `pressure` (`NDArray[np.float64]`) — Box pressures in pascals. Shape: ``(n_boxes,)``.
- `saturation_ratio` (`NDArray[np.float64]`) — Per-box, per-species saturation ratios. Shape: ``(n_boxes, n_species)``.

**Raises**

- `ValueError` — If any field has invalid dimensionality, shape,
finiteness, or physical bounds.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L21-L215)

### particula.gas.EnvironmentData.pressure

*attribute* · *instance attribute*

```python
pressure: NDArray[np.float64] = self._coerce_float64_array(pressure, field_name='pressure')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L63-L66)

### particula.gas.EnvironmentData.saturation_ratio

*attribute* · *instance attribute*

```python
saturation_ratio: NDArray[np.float64] = self._coerce_float64_array(saturation_ratio, field_name='saturation_ratio')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L67-L70)

### particula.gas.EnvironmentData.temperature

*attribute* · *instance attribute*

```python
temperature: NDArray[np.float64] = self._coerce_float64_array(temperature, field_name='temperature')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L59-L62)

### particula.gas.EnvironmentData.n_boxes

*property*

```python
n_boxes: int
```

Return the number of simulation boxes.

**Returns**

- (`int`) — Number of boxes represented by ``temperature.shape[0]``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L86-L92)

### particula.gas.EnvironmentData.__init__

*method*

```python
def __init__(temperature: ArrayLike, pressure: ArrayLike, saturation_ratio: ArrayLike) -> None
```

Construct and normalize per-box environment data.

**Parameters**

- `temperature` (`ArrayLike`) — Box temperatures in kelvin with shape ``(n_boxes,)``.
- `pressure` (`ArrayLike`) — Box pressures in pascals with shape ``(n_boxes,)``.
- `saturation_ratio` (`ArrayLike`) — Per-box, per-species saturation ratios with shape ``(n_boxes, n_species)``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L45-L71)

### particula.gas.EnvironmentData.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate the normalized field contract.

**Raises**

- `ValueError` — If any field fails dimensionality, shape, finiteness,
or physical-bound validation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L73-L83)

### particula.gas.EnvironmentData._coerce_float64_array

*method* · *staticmethod*

```python
def _coerce_float64_array(values: object, *, field_name: str) -> NDArray[np.float64]
```

Convert array-like input to a ``np.float64`` NumPy array.

**Parameters**

- `values` (`object`) — Array-like input supplied for a dataclass field.
- `field_name` (`str`) — Name of the field being coerced for error reporting.

**Returns**

- (`NDArray[np.float64]`) — NumPy array with ``np.float64`` dtype.

**Raises**

- `ValueError` — If ``values`` cannot be converted to a float64 array.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L107-L130)

### particula.gas.EnvironmentData._validate_dimensionality

*method*

```python
def _validate_dimensionality() -> None
```

Validate per-field dimensionality requirements.

**Raises**

- `ValueError` — If ``temperature`` or ``pressure`` is not 1D, or if
``saturation_ratio`` is not 2D.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L132-L154)

### particula.gas.EnvironmentData._validate_finite_values

*method*

```python
def _validate_finite_values() -> None
```

Require all numeric fields to be finite.

**Raises**

- `ValueError` — If any field contains ``NaN`` or infinite values.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L186-L199)

### particula.gas.EnvironmentData._validate_physical_bounds

*method*

```python
def _validate_physical_bounds() -> None
```

Require positive thermodynamic state and nonnegative saturation.

**Raises**

- `ValueError` — If temperature or pressure is nonpositive, or if any
saturation ratio is negative.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L201-L215)

### particula.gas.EnvironmentData._validate_shapes

*method*

```python
def _validate_shapes() -> None
```

Validate shared box-count consistency after ndim checks.

**Raises**

- `ValueError` — If no boxes are provided, or if ``pressure`` or
``saturation_ratio`` does not share the same leading box
dimension as ``temperature``.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L156-L184)

### particula.gas.EnvironmentData.copy

*method*

```python
def copy() -> EnvironmentData
```

Create an independent copy of the environment state.

**Returns**

- `New` (`EnvironmentData`) — class:`EnvironmentData` instance with copied temperature,
- (`EnvironmentData`) — pressure, and saturation-ratio arrays.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/environment_data.py#L94-L105)

## particula.gas.GasData

*class* · *dataclass*

```python
class GasData
```

Re-exported from: `particula.gas.gas_data`

Batched gas species data container for multi-box simulations.

Simple data container with batch dimension built-in. Concentration
arrays have shape (n_boxes, n_species) to support multi-box CFD.
Single-box simulations use n_boxes=1.

``GasData`` is also the CPU-side owner of ordered species names. GPU
transfer helpers exclude ``name``, convert ``partitioning`` as
``bool → int32 → bool`` with binary ``0``/``1`` restore requirements, and
intentionally exclude GPU-only helper state such as vapor pressure from
restored CPU data. ``from_warp_gas_data()`` can rebuild names only from
caller-managed ordered names or placeholder names; it cannot recover
semantic species identity from GPU data alone.

This is NOT a frozen dataclass - concentrations can be updated in place
for performance. Use copy() if immutability is needed.

**Attributes**

- `name` (`list[str]`) — Species names. List of strings, length n_species.
- `molar_mass` (`NDArray[np.float64]`) — Molar masses in kg/mol. Shape: (n_species,)
- `concentration` (`NDArray[np.float64]`) — Mass concentrations in kg/m^3. Shape: (n_boxes, n_species)
- `partitioning` (`NDArray[np.bool_]`) — Whether each species can partition to particles. Shape: (n_species,) - shared across boxes

**Raises**

- `ValueError` — If array shapes are inconsistent or species list is empty.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L41-L142)

### particula.gas.GasData.concentration

*attribute* · *instance attribute*

```python
concentration: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L74-L74)

### particula.gas.GasData.molar_mass

*attribute* · *instance attribute*

```python
molar_mass: NDArray[np.float64]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L73-L73)

### particula.gas.GasData.name

*attribute* · *instance attribute*

```python
name: list[str]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L72-L72)

### particula.gas.GasData.partitioning

*attribute* · *instance attribute*

```python
partitioning: NDArray[np.bool_]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L75-L75)

### particula.gas.GasData.n_boxes

*property*

```python
n_boxes: int
```

Number of simulation boxes (batch dimension).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L126-L128)

### particula.gas.GasData.n_species

*property*

```python
n_species: int
```

Number of gas species.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L131-L133)

### particula.gas.GasData.__init__

*method*

```python
def __init__(name: list[str], molar_mass: NDArray[np.float64], concentration: NDArray[np.float64], partitioning: NDArray[np.bool_]) -> None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L0-L0)

### particula.gas.GasData.__post_init__

*method*

```python
def __post_init__() -> None
```

Validate array shapes are consistent and enforce boolean mask.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L77-L123)

### particula.gas.GasData.copy

*method*

```python
def copy() -> GasData
```

Create a deep copy of this GasData.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data.py#L135-L142)

## particula.gas.GasDataBuilder

*class*

```python
class GasDataBuilder
```

Re-exported from: `particula.gas.gas_data_builder`

Fluent builder that prepares arrays for ``GasData``.

Each setter performs unit conversion and ensures correct dtypes.
Batch dimensions are inserted automatically when 1D concentration is
provided.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L45-L206)

### particula.gas.GasDataBuilder._concentration

*attribute* · *instance attribute*

```python
_concentration: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L57-L57)

### particula.gas.GasDataBuilder._molar_mass

*attribute* · *instance attribute*

```python
_molar_mass: Optional[NDArray[np.float64]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L56-L56)

### particula.gas.GasDataBuilder._n_boxes

*attribute* · *instance attribute*

```python
_n_boxes: Optional[int] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L59-L59)

### particula.gas.GasDataBuilder._names

*attribute* · *instance attribute*

```python
_names: Optional[list[str]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L55-L55)

### particula.gas.GasDataBuilder._partitioning

*attribute* · *instance attribute*

```python
_partitioning: Optional[NDArray[np.bool_]] = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L58-L58)

### particula.gas.GasDataBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize empty builder state.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L53-L59)

### particula.gas.GasDataBuilder.build

*method*

```python
def build() -> GasData
```

Build and return validated GasData.

**Returns**

- (`GasData`) — A validated GasData instance.

**Raises**

- `ValueError` — If required fields are missing or invalid.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L176-L206)

### particula.gas.GasDataBuilder.set_concentration

*method*

```python
def set_concentration(concentration: Union[list[float], NDArray[np.float64]], units: str = 'kg/m^3') -> 'GasDataBuilder'
```

Set concentrations with unit conversion and auto batch dimension.

**Parameters**

- `concentration` (`Union[list[float], NDArray[np.float64]]`) — Concentration values. If 1D (n_species,), batch dimension added. If 2D (n_boxes, n_species), used as-is.
- `units` (`str`) (default: `'kg/m^3'`) — Units of the provided concentration. Supported: kg/m^3, g/m^3.

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When any concentration is negative or wrong dimension.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L105-L140)

### particula.gas.GasDataBuilder.set_molar_mass

*method*

```python
def set_molar_mass(molar_mass: Union[list[float], NDArray[np.float64]], units: str = 'kg/mol') -> 'GasDataBuilder'
```

Set molar masses with optional unit conversion.

**Parameters**

- `molar_mass` (`Union[list[float], NDArray[np.float64]]`) — Molar mass values shaped (n_species,).
- `units` (`str`) (default: `'kg/mol'`) — Units of the provided values. Supported: kg/mol, g/mol.

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When molar_mass is not 1D or any value is non-positive.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L73-L103)

### particula.gas.GasDataBuilder.set_n_boxes

*method*

```python
def set_n_boxes(n_boxes: int) -> 'GasDataBuilder'
```

Set number of boxes for broadcasting 1D concentration.

**Parameters**

- `n_boxes` (`int`) — Number of simulation boxes.

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L164-L174)

### particula.gas.GasDataBuilder.set_names

*method*

```python
def set_names(names: list[str]) -> 'GasDataBuilder'
```

Set species names.

**Parameters**

- `names` (`list[str]`) — List of species names.

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L61-L71)

### particula.gas.GasDataBuilder.set_partitioning

*method*

```python
def set_partitioning(partitioning: Union[list[bool], NDArray[np.bool_]]) -> 'GasDataBuilder'
```

Set whether each species can partition.

**Parameters**

- `partitioning` (`Union[list[bool], NDArray[np.bool_]]`) — Boolean array indicating which species partition. Shape: (n_species,).

**Returns**

- (`'GasDataBuilder'`) — Self for fluent chaining.

**Raises**

- `ValueError` — When partitioning is not 1D.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/gas_data_builder.py#L142-L162)

## particula.gas.LatentHeatFactory

*class*

```python
class LatentHeatFactory(StrategyFactoryABC[LatentHeatBuilderType, LatentHeatStrategyType])
```

Bases: ``

Re-exported from: `particula.gas.latent_heat_factories`

Factory for latent heat strategies.

The factory builds constant, linear, or power-law latent heat strategies
by delegating to the corresponding builder class.

**Examples**

```pycon
>>> from particula.gas import LatentHeatFactory
>>> factory = LatentHeatFactory()
>>> factory.get_strategy(
...     "constant",
...     {
...         "latent_heat_ref": 2.26e6,
...         "latent_heat_ref_units": "J/kg",
...     },
... ).latent_heat(300.0)
2260000.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_factories.py#L30-L67)

### particula.gas.LatentHeatFactory.get_builders

*method*

```python
def get_builders() -> dict[str, LatentHeatBuilderType]
```

Return latent heat builders keyed by strategy name.

**Returns**

- (`dict[str, LatentHeatBuilderType]`) — Builder instances keyed by strategy type.

**Examples**

```pycon
>>> from particula.gas import LatentHeatFactory
>>> builders = LatentHeatFactory().get_builders()
>>> isinstance(builders["constant"], ConstantLatentHeatBuilder)
True
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_factories.py#L51-L67)

## particula.gas.LatentHeatStrategy

*class*

```python
class LatentHeatStrategy(ABC)
```

Bases: `ABC`

Re-exported from: `particula.gas.latent_heat_strategies`

Abstract base class for latent heat calculations.

Subclasses implement :meth:`latent_heat` to return the latent heat of
vaporization in J/kg for scalar or array temperature inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L16-L34)

### particula.gas.LatentHeatStrategy.latent_heat

*method* · *abstract*

```python
def latent_heat(temperature: float | NDArray[np.float64]) -> float | NDArray[np.float64]
```

Return latent heat of vaporization for the given temperature.

**Parameters**

- `temperature` (`float | NDArray[np.float64]`) — Temperature in Kelvin.

**Returns**

- (`float | NDArray[np.float64]`) — Latent heat of vaporization in J/kg with shape matching the input.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L23-L34)

## particula.gas.LinearLatentHeat

*class*

```python
class LinearLatentHeat(LatentHeatStrategy)
```

Bases: `particula.gas.latent_heat_strategies.LatentHeatStrategy`

Re-exported from: `particula.gas.latent_heat_strategies`

Latent heat strategy using a linear temperature dependence.

Computes a linear relation for each temperature:
:math:`L(T) = L_{ref} - slope * (T - T_{ref})`.

**Attributes**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg at the reference temperature.
- `slope` (`float`) — Linear temperature slope in J/(kg·K).
- `temperature_ref` (`float`) — Reference temperature in Kelvin.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L87-L146)

### particula.gas.LinearLatentHeat.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float = latent_heat_ref
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L120-L120)

### particula.gas.LinearLatentHeat.slope

*attribute* · *instance attribute*

```python
slope: float = slope
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L121-L121)

### particula.gas.LinearLatentHeat.temperature_ref

*attribute* · *instance attribute*

```python
temperature_ref: float = temperature_ref
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L122-L122)

### particula.gas.LinearLatentHeat.__init__

*method*

```python
def __init__(latent_heat_ref: float, slope: float, temperature_ref: float) -> None
```

Initialize the linear latent heat strategy.

**Parameters**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg.
- `slope` (`float`) — Linear slope in J/(kg·K).
- `temperature_ref` (`float`) — Reference temperature in Kelvin.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L100-L122)

### particula.gas.LinearLatentHeat.latent_heat

*method*

```python
def latent_heat(temperature: float | NDArray[np.float64]) -> float | NDArray[np.float64]
```

Return the latent heat for the given temperature.

Uses the linear relation
:math:`L(T) = L_{ref} - slope * (T - T_{ref})`.

**Parameters**

- `temperature` (`float | NDArray[np.float64]`) — Temperature in Kelvin.

**Returns**

- (`float | NDArray[np.float64]`) — Latent heat in J/kg. Returns a scalar for scalar input or an array
- (`float | NDArray[np.float64]`) — matching the input shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L124-L146)

## particula.gas.LinearLatentHeatBuilder

*class*

```python
class LinearLatentHeatBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.gas.latent_heat_builders`

Build LinearLatentHeat strategies with validated inputs.

Requires a reference latent heat, linear slope, and reference temperature.

**Attributes**

- `latent_heat_ref` (`float | None`) — Reference latent heat in J/kg once set.
- `slope` (`float | None`) — Linear slope in J/(kg*K) once set.
- `temperature_ref` (`float | None`) — Reference temperature in K once set.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L111-L235)

### particula.gas.LinearLatentHeatBuilder.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L126-L126)

### particula.gas.LinearLatentHeatBuilder.slope

*attribute* · *instance attribute*

```python
slope: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L127-L127)

### particula.gas.LinearLatentHeatBuilder.temperature_ref

*attribute* · *instance attribute*

```python
temperature_ref: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L128-L128)

### particula.gas.LinearLatentHeatBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.LinearLatentHeatBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the linear latent heat builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L122-L128)

### particula.gas.LinearLatentHeatBuilder.build

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

### particula.gas.LinearLatentHeatBuilder.set_latent_heat_ref

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

### particula.gas.LinearLatentHeatBuilder.set_slope

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

### particula.gas.LinearLatentHeatBuilder.set_temperature_ref

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

### particula.gas.LinearLatentHeatBuilder.check_keys

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

### particula.gas.LinearLatentHeatBuilder.pre_build_check

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

### particula.gas.LinearLatentHeatBuilder.set_parameters

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

## particula.gas.LiquidClausiusHybridStrategy

*class*

```python
class LiquidClausiusHybridStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Re-exported from: `particula.gas.vapor_pressure_strategies`

Hybrid strategy blending liquid and Clausius–Clapeyron methods.

This strategy uses the 5-term liquid polynomial at low temperature and
gradually transitions to the Clausius–Clapeyron relation above the
boiling point. A logistic weight provides a smooth changeover.

Methods:
- pure_vapor_pressure: Compute the blended vapor pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L600-L661)

### particula.gas.LiquidClausiusHybridStrategy.boiling_point

*attribute* · *instance attribute*

```python
boiling_point = boiling_point
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L636-L636)

### particula.gas.LiquidClausiusHybridStrategy.clausius_strategy

*attribute* · *instance attribute*

```python
clausius_strategy = ClausiusClapeyronStrategy(latent_heat, temperature_initial, pressure_initial)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L633-L635)

### particula.gas.LiquidClausiusHybridStrategy.liquid_strategy

*attribute* · *instance attribute*

```python
liquid_strategy = ArblasterLiquidVaporPressureStrategy(coefficients)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L630-L632)

### particula.gas.LiquidClausiusHybridStrategy.transition_width

*attribute* · *instance attribute*

```python
transition_width = transition_width
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L637-L637)

### particula.gas.LiquidClausiusHybridStrategy.__init__

*method*

```python
def __init__(coefficients: tuple[float, float, float, float, float], latent_heat: float, temperature_initial: float, pressure_initial: float, boiling_point: float, transition_width: float = 10.0) -> None
```

Initialize the hybrid vapor pressure strategy.

**Parameters**

- `coefficients` (`tuple[float, float, float, float, float]`) — Polynomial coefficients ``(A, B, C, D, E)``.
- `latent_heat` (`float`) — Latent heat of vaporization in J/mol.
- `temperature_initial` (`float`) — Reference temperature in Kelvin.
- `pressure_initial` (`float`) — Reference pressure in Pascals.
- `boiling_point` (`float`) — Temperature where weighting is 0.5.
- `transition_width` (`float`) (default: `10.0`) — Width of the logistic transition in Kelvin.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L611-L637)

### particula.gas.LiquidClausiusHybridStrategy.pure_vapor_pressure

*method*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Smoothly combine liquid and Clausius–Clapeyron approaches.

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure in Pascals.

**Examples**

``` py title="Hybrid vapor pressure"
vp = strategy.pure_vapor_pressure(boiling_point + 50.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L639-L661)

### particula.gas.LiquidClausiusHybridStrategy.concentration

*method*

```python
def concentration(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the concentration of the gas at a given pressure and
temperature.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Pressure in Pascals.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The concentration of the gas in kg/m^3.

**Examples**

``` py title="Concentration Calculation"
concentration = strategy.concentration(
    partial_pressure=101325,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L87-L115)

### particula.gas.LiquidClausiusHybridStrategy.partial_pressure

*method*

```python
def partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate partial pressure from concentration, molar mass, T.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals.

**Examples**

``` py title="Partial Pressure Calculation"
partial_pressure = strategy.partial_pressure(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L60-L85)

### particula.gas.LiquidClausiusHybridStrategy.saturation_concentration

*method*

```python
def saturation_concentration(molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation concentration of the gas at a given
temperature.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation concentration of the gas in kg/m^3.

**Examples**

``` py title="Saturation Concentration Calculation"
saturation_concentration = strategy.saturation_concentration(
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L147-L172)

### particula.gas.LiquidClausiusHybridStrategy.saturation_ratio

*method*

```python
def saturation_ratio(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation ratio of the gas from its concentration and
molar mass at a given temperature.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation ratio of the gas.

**Examples**

``` py title="Saturation Ratio Calculation"
saturation_ratio = strategy.saturation_ratio(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L117-L145)

## particula.gas.PowerLawLatentHeat

*class*

```python
class PowerLawLatentHeat(LatentHeatStrategy)
```

Bases: `particula.gas.latent_heat_strategies.LatentHeatStrategy`

Re-exported from: `particula.gas.latent_heat_strategies`

Latent heat strategy using a power-law temperature dependence.

Computes a power-law relation:
:math:`L(T) = L_{ref} * (1 - T / T_c)^{beta}`.
The ratio is clipped to [0, 1] to enforce non-negative latent heat values.

**Attributes**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg.
- `critical_temperature` (`float`) — Critical temperature in Kelvin.
- `beta` (`float`) — Power-law exponent (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L149-L219)

### particula.gas.PowerLawLatentHeat.beta

*attribute* · *instance attribute*

```python
beta: float = beta
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L184-L184)

### particula.gas.PowerLawLatentHeat.critical_temperature

*attribute* · *instance attribute*

```python
critical_temperature: float = critical_temperature
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L183-L183)

### particula.gas.PowerLawLatentHeat.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float = latent_heat_ref
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L182-L182)

### particula.gas.PowerLawLatentHeat.__init__

*method*

```python
def __init__(latent_heat_ref: float, critical_temperature: float, beta: float) -> None
```

Initialize the power-law latent heat strategy.

**Parameters**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg.
- `critical_temperature` (`float`) — Critical temperature in Kelvin.
- `beta` (`float`) — Power-law exponent (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L162-L184)

### particula.gas.PowerLawLatentHeat.latent_heat

*method*

```python
def latent_heat(temperature: float | NDArray[np.float64]) -> float | NDArray[np.float64]
```

Return the latent heat for the given temperature.

Temperatures are normalized by the critical temperature and clipped
to the range [0, 1], ensuring latent heat values stay non-negative.

**Parameters**

- `temperature` (`float | NDArray[np.float64]`) — Temperature in Kelvin.

**Returns**

- (`float | NDArray[np.float64]`) — Latent heat in J/kg. Returns a scalar for scalar input or an array
- (`float | NDArray[np.float64]`) — matching the input shape.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L186-L219)

## particula.gas.PowerLawLatentHeatBuilder

*class*

```python
class PowerLawLatentHeatBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.gas.latent_heat_builders`

Build PowerLawLatentHeat strategies with validated inputs.

Requires a reference latent heat, critical temperature, and exponent.

**Attributes**

- `latent_heat_ref` (`float | None`) — Reference latent heat in J/kg once set.
- `critical_temperature` (`float | None`) — Critical temperature in K once set.
- `beta` (`float | None`) — Power-law exponent (dimensionless) once set.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L238-L366)

### particula.gas.PowerLawLatentHeatBuilder.beta

*attribute* · *instance attribute*

```python
beta: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L259-L259)

### particula.gas.PowerLawLatentHeatBuilder.critical_temperature

*attribute* · *instance attribute*

```python
critical_temperature: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L258-L258)

### particula.gas.PowerLawLatentHeatBuilder.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L257-L257)

### particula.gas.PowerLawLatentHeatBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.PowerLawLatentHeatBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the power-law latent heat builder.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_builders.py#L249-L259)

### particula.gas.PowerLawLatentHeatBuilder.build

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

### particula.gas.PowerLawLatentHeatBuilder.set_beta

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

### particula.gas.PowerLawLatentHeatBuilder.set_critical_temperature

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

### particula.gas.PowerLawLatentHeatBuilder.set_latent_heat_ref

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

### particula.gas.PowerLawLatentHeatBuilder.check_keys

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

### particula.gas.PowerLawLatentHeatBuilder.pre_build_check

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

### particula.gas.PowerLawLatentHeatBuilder.set_parameters

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

## particula.gas.SaturationConcentrationVaporPressureBuilder

*class*

```python
class SaturationConcentrationVaporPressureBuilder(BuilderABC, BuilderMolarMassMixin, BuilderTemperatureMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderMolarMassMixin`, `particula.builder_mixin.BuilderTemperatureMixin`

Re-exported from: `particula.gas.vapor_pressure_builders`

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

### particula.gas.SaturationConcentrationVaporPressureBuilder.molar_mass

*attribute* · *instance attribute*

```python
molar_mass = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L386-L386)

### particula.gas.SaturationConcentrationVaporPressureBuilder.saturation_concentration

*attribute* · *instance attribute*

```python
saturation_concentration = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L385-L385)

### particula.gas.SaturationConcentrationVaporPressureBuilder.temperature

*attribute* · *instance attribute*

```python
temperature = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L387-L387)

### particula.gas.SaturationConcentrationVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.SaturationConcentrationVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Saturation Concentration vapor pressure builder.

Sets up the builder with required parameters for creating a
ConstantVaporPressureStrategy from saturation concentration,
molar mass, and temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L370-L387)

### particula.gas.SaturationConcentrationVaporPressureBuilder.build

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

### particula.gas.SaturationConcentrationVaporPressureBuilder.set_saturation_concentration

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

### particula.gas.SaturationConcentrationVaporPressureBuilder.check_keys

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

### particula.gas.SaturationConcentrationVaporPressureBuilder.pre_build_check

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

### particula.gas.SaturationConcentrationVaporPressureBuilder.set_parameters

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

### particula.gas.SaturationConcentrationVaporPressureBuilder.set_molar_mass

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

### particula.gas.SaturationConcentrationVaporPressureBuilder.set_temperature

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

## particula.gas.TableVaporPressureBuilder

*class*

```python
class TableVaporPressureBuilder(BuilderABC, BuilderTemperatureTableMixin)
```

Bases: `particula.abc_builder.BuilderABC`, `particula.builder_mixin.BuilderTemperatureTableMixin`

Re-exported from: `particula.gas.vapor_pressure_builders`

Builder for TableVaporPressureStrategy.

Allows setting lookup tables of vapor pressure and temperature, handling
optional unit conversion before constructing the strategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L472-L517)

### particula.gas.TableVaporPressureBuilder.vapor_pressure_table

*attribute* · *instance attribute*

```python
vapor_pressure_table: NDArray[np.float64] | None = None
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L492-L492)

### particula.gas.TableVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.TableVaporPressureBuilder.temperature_table

*attribute* · *instance attribute*

```python
temperature_table: Optional[NDArray[np.float64]] = None
```

Inherited from: `particula.builder_mixin.BuilderTemperatureTableMixin`

[View source](https://github.com/uncscode/particula/blob/main/particula/builder_mixin.py#L426-L426)

### particula.gas.TableVaporPressureBuilder.__init__

*method*

```python
def __init__() -> None
```

Initialize the Table vapor pressure builder.

Sets up the builder with required parameters for creating a
TableVaporPressureStrategy from lookup tables of vapor pressure
and temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L482-L492)

### particula.gas.TableVaporPressureBuilder.build

*method*

```python
def build() -> TableVaporPressureStrategy
```

Validate and construct a TableVaporPressureStrategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L510-L517)

### particula.gas.TableVaporPressureBuilder.set_vapor_pressure_table

*method*

```python
def set_vapor_pressure_table(vapor_pressure_table: NDArray[np.float64], vapor_pressure_table_units: str = 'Pa') -> TableVaporPressureBuilder
```

Set the vapor pressure lookup table.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L494-L508)

### particula.gas.TableVaporPressureBuilder.check_keys

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

### particula.gas.TableVaporPressureBuilder.pre_build_check

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

### particula.gas.TableVaporPressureBuilder.set_parameters

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

### particula.gas.TableVaporPressureBuilder.set_temperature_table

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

## particula.gas.TableVaporPressureStrategy

*class*

```python
class TableVaporPressureStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Re-exported from: `particula.gas.vapor_pressure_strategies`

Vapor pressure strategy using interpolation of a lookup table.

This strategy accepts a set of temperatures and corresponding vapor
pressures and linearly interpolates between the points to determine
the vapor pressure at arbitrary temperatures.

**Attributes**

- `vapor_pressures` — NDArray[np.float64] Vapor pressure values in Pascals.
- `temperatures` — NDArray[np.float64] Temperatures in Kelvin corresponding to the vapor pressures.

Methods:
- partial_pressure: Compute partial pressure from concentration.
- concentration: Compute concentration from partial pressure.
- saturation_ratio: Compute ratio of partial pressure to saturation
  pressure.
- saturation_concentration: Compute concentration at saturation pressure.
- pure_vapor_pressure: Interpolate the vapor pressure for the
  given temperature.

**Examples**

```py title="Table Vapor Pressure Example"
import numpy as np
from particula.gas import TableVaporPressureStrategy

table_vp = [100.0, 200.0, 300.0, 400.0]  # Pa
table_t = [270.0, 280.0, 290.0, 300.0]   # K
strategy = TableVaporPressureStrategy(
    vapor_pressures=table_vp,
    temperatures=table_t
)
vp = strategy.pure_vapor_pressure(295.0)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L469-L540)

### particula.gas.TableVaporPressureStrategy.temperatures

*attribute* · *instance attribute*

```python
temperatures = table_temps[idx]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L528-L528)

### particula.gas.TableVaporPressureStrategy.vapor_pressures

*attribute* · *instance attribute*

```python
vapor_pressures = table_pressures[idx]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L527-L527)

### particula.gas.TableVaporPressureStrategy.__init__

*method*

```python
def __init__(vapor_pressures: NDArray[np.float64], temperatures: NDArray[np.float64]) -> None
```

Initialize the Table Vapor Pressure strategy.

**Parameters**

- `vapor_pressures` (`NDArray[np.float64]`) — Array of vapor pressure values in Pascals.
- `temperatures` (`NDArray[np.float64]`) — Array of corresponding temperatures in Kelvin.

**Raises**

- `ValueError` — If temperature and pressure tables differ in length.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L506-L528)

### particula.gas.TableVaporPressureStrategy.pure_vapor_pressure

*method*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Interpolate the vapor pressure for the given temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L530-L540)

### particula.gas.TableVaporPressureStrategy.concentration

*method*

```python
def concentration(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the concentration of the gas at a given pressure and
temperature.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Pressure in Pascals.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The concentration of the gas in kg/m^3.

**Examples**

``` py title="Concentration Calculation"
concentration = strategy.concentration(
    partial_pressure=101325,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L87-L115)

### particula.gas.TableVaporPressureStrategy.partial_pressure

*method*

```python
def partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate partial pressure from concentration, molar mass, T.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals.

**Examples**

``` py title="Partial Pressure Calculation"
partial_pressure = strategy.partial_pressure(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L60-L85)

### particula.gas.TableVaporPressureStrategy.saturation_concentration

*method*

```python
def saturation_concentration(molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation concentration of the gas at a given
temperature.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation concentration of the gas in kg/m^3.

**Examples**

``` py title="Saturation Concentration Calculation"
saturation_concentration = strategy.saturation_concentration(
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L147-L172)

### particula.gas.TableVaporPressureStrategy.saturation_ratio

*method*

```python
def saturation_ratio(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation ratio of the gas from its concentration and
molar mass at a given temperature.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation ratio of the gas.

**Examples**

``` py title="Saturation Ratio Calculation"
saturation_ratio = strategy.saturation_ratio(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L117-L145)

## particula.gas.VaporPressureFactory

*class*

```python
class VaporPressureFactory(StrategyFactoryABC[Union[ConstantVaporPressureBuilder, AntoineVaporPressureBuilder, ClausiusClapeyronVaporPressureBuilder, SaturationConcentrationVaporPressureBuilder, WaterBuckVaporPressureBuilder, TableVaporPressureBuilder], Union[ConstantVaporPressureStrategy, AntoineVaporPressureStrategy, ClausiusClapeyronStrategy, WaterBuckStrategy, TableVaporPressureStrategy]])
```

Bases: ``

Re-exported from: `particula.gas.vapor_pressure_factories`

Factory class to create vapor pressure strategy
builders.

This class provides a way to generate multiple vapor pressure calculation
strategies (e.g., constant, Antoine, Clausius-Clapeyron, or Water Buck) by
commissioning the appropriate builder. It is useful for scenarios requiring
a flexible way to switch or extend vapor pressure calculation methods.

- get_builders : Returns the mapping of strategy types to builder
  instances.
- get_strategy : Returns the selected vapor pressure strategy,
  given a strategy type and parameters.

**Examples**

```py title="Example VaporPressureFactory usage"
import particula as par

factory = par.gas.VaporPressureFactory()
# Create a constant vapor pressure strategy:
strategy = factory.get_strategy(
    "constant", {"constant_vapor_pressure": 101325.0}
)
# strategy is an instance of ConstantVaporPressureStrategy
```

> **References**
>
> - "Vapor Pressure,"
> [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_factories.py#L25-L104)

### particula.gas.VaporPressureFactory.get_builders

*method*

```python
def get_builders()
```

Return a dictionary mapping strategy types to builder instances.

**Returns**

- `result` (`dict`) — Available builders. - "constant": ConstantVaporPressureBuilder - "antoine": AntoineVaporPressureBuilder - "clausius_clapeyron": ClausiusClapeyronVaporPressureBuilder - "saturation_concentration": SaturationConcentration   VaporPressureBuilder - "water_buck": WaterBuckVaporPressureBuilder

**Examples**

```py
import particula as par
builders_dict = par.gas.VaporPressureFactory().get_builders()
builder = builders_dict["constant"]
# builder is an instance of ConstantBuilder
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_factories.py#L75-L104)

## particula.gas.WaterBuckStrategy

*class*

```python
class WaterBuckStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Re-exported from: `particula.gas.vapor_pressure_strategies`

Vapor pressure strategy using the Buck equation for water.

This class computes water vapor pressure using the Buck equation, an
empirically derived correlation often applied in meteorology to determine
the saturation vapor pressure of water.

Methods:
- partial_pressure: Compute partial pressure from concentration.
- concentration: Compute concentration from partial pressure.
- saturation_ratio: Compute ratio of partial pressure to saturation
  pressure.
- saturation_concentration: Compute concentration at saturation pressure.
- pure_vapor_pressure: Computes water vapor pressure from the Buck
  equation.

**Examples**

```py title="Water Buck Vapor Pressure Example"
strategy = WaterBuckStrategy()
vp = strategy.pure_vapor_pressure(temperature=298.15)
# Returns water vapor pressure in Pascals
```

> **References**
>
> - A. L. Buck, "New Equations for Computing Vapor Pressure...",
>   J. Appl. Meteor. Climatol. 20(12), 1527–1532 (1981).
> - https://en.wikipedia.org/wiki/Arden_Buck_equation

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L413-L466)

### particula.gas.WaterBuckStrategy.pure_vapor_pressure

*method*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate vapor pressure using the Buck equation for water vapor.

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure in Pascals.

**Examples**

``` py title="Water Buck Vapor Pressure Calculation"
vapor_pressure = strategy.pure_vapor_pressure(
    temperature=300
)
```

> **References**
>
> - Buck, A. L., 1981: New Equations for Computing Vapor Pressure and
>   Enhancement Factor. J. Appl. Meteor. Climatol., 20, 1527-1532,
>   https://doi.org/10.1175/1520-0450(1981)020<1527:NEFCVP>2.0.CO;2.
> - https://en.wikipedia.org/wiki/Arden_Buck_equation

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L442-L466)

### particula.gas.WaterBuckStrategy.concentration

*method*

```python
def concentration(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the concentration of the gas at a given pressure and
temperature.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Pressure in Pascals.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The concentration of the gas in kg/m^3.

**Examples**

``` py title="Concentration Calculation"
concentration = strategy.concentration(
    partial_pressure=101325,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L87-L115)

### particula.gas.WaterBuckStrategy.partial_pressure

*method*

```python
def partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate partial pressure from concentration, molar mass, T.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals.

**Examples**

``` py title="Partial Pressure Calculation"
partial_pressure = strategy.partial_pressure(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L60-L85)

### particula.gas.WaterBuckStrategy.saturation_concentration

*method*

```python
def saturation_concentration(molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation concentration of the gas at a given
temperature.

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation concentration of the gas in kg/m^3.

**Examples**

``` py title="Saturation Concentration Calculation"
saturation_concentration = strategy.saturation_concentration(
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L147-L172)

### particula.gas.WaterBuckStrategy.saturation_ratio

*method*

```python
def saturation_ratio(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Calculate the saturation ratio of the gas from its concentration and
molar mass at a given temperature.

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m^3.
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The saturation ratio of the gas.

**Examples**

``` py title="Saturation Ratio Calculation"
saturation_ratio = strategy.saturation_ratio(
    concentration=5.0,
    molar_mass=18.01528,
    temperature=298.15
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L117-L145)

## particula.gas.WaterBuckVaporPressureBuilder

*class*

```python
class WaterBuckVaporPressureBuilder(BuilderABC)
```

Bases: `particula.abc_builder.BuilderABC`

Re-exported from: `particula.gas.vapor_pressure_builders`

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

### particula.gas.WaterBuckVaporPressureBuilder.required_parameters

*attribute* · *instance attribute*

```python
required_parameters = required_parameters or []
```

Inherited from: `particula.abc_builder.BuilderABC`

[View source](https://github.com/uncscode/particula/blob/main/particula/abc_builder.py#L51-L51)

### particula.gas.WaterBuckVaporPressureBuilder.__init__

*method*

```python
def __init__()
```

Initialize the Water Buck vapor pressure builder.

Sets up the builder for creating a WaterBuckStrategy.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L456-L461)

### particula.gas.WaterBuckVaporPressureBuilder.build

*method*

```python
def build() -> WaterBuckStrategy
```

Build and return a WaterBuckStrategy object.

**Returns**

- (`WaterBuckStrategy`) — Configured for water-specific Buck vapor pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_builders.py#L463-L469)

### particula.gas.WaterBuckVaporPressureBuilder.check_keys

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

### particula.gas.WaterBuckVaporPressureBuilder.pre_build_check

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

### particula.gas.WaterBuckVaporPressureBuilder.set_parameters

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

## particula.gas.get_antoine_vapor_pressure

*function*

```python
def get_antoine_vapor_pressure(a: Union[float, NDArray[np.float64]], b: Union[float, NDArray[np.float64]], c: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.vapor_pressure_module`

Calculate vapor pressure using the Antoine equation.

The Antoine equation relates the logarithm of vapor pressure to
temperature for a pure substance.

- P = 10^(a - b / (T - c)) × 133.322
    - P is Vapor pressure [Pa].
    - a, b, c is Antoine equation parameters (dimensionless).
    - T is Temperature [K].

**Parameters**

- `a` (`Union[float, NDArray[np.float64]]`) — Antoine parameter a (dimensionless).
- `b` (`Union[float, NDArray[np.float64]]`) — Antoine parameter b (dimensionless).
- `c` (`Union[float, NDArray[np.float64]]`) — Antoine parameter c (dimensionless).
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin [K].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure in Pascals [Pa].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_antoine_vapor_pressure(
    8.07131, 1730.63, 233.426, 373.15
)
# Output: ~101325 Pa (roughly 1 atm)
```

> **References**
>
> - https://en.wikipedia.org/wiki/Antoine_equation
> - Kelvin conversion details:
>   https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118135341.app1

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/vapor_pressure_module.py#L12-L61)

## particula.gas.get_buck_vapor_pressure

*function*

```python
def get_buck_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.vapor_pressure_module`

Calculate vapor pressure using the Buck equation for water vapor.

Uses separate empirical formulas below 0 °C and above 0 °C to compute
water vapor pressure.

- For T < 0 °C, as
    p = 6.1115 × exp( (23.036 - T/333.7) × T / (279.82 + T ) ) × 100
- For T ≥ 0 °C, as
    p = 6.1121 × exp( (18.678 - T/234.5) × T / (257.14 + T ) ) × 100
    - p is Vapor pressure [Pa].
    - T is Temperature in Celsius [°C] (converted internally from Kelvin).

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin [K].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Vapor pressure in Pascals [Pa].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_buck_vapor_pressure(273.15)
# Output: ~611 Pa (around ice point)
```

> **References**
>
> - Buck, A. L., (1981)
> - https://en.wikipedia.org/wiki/Arden_Buck_equation

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/vapor_pressure_module.py#L121-L172)

## particula.gas.get_clausius_clapeyron_vapor_pressure

*function*

```python
def get_clausius_clapeyron_vapor_pressure(latent_heat: Union[float, NDArray[np.float64]], temperature_initial: Union[float, NDArray[np.float64]], pressure_initial: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]], gas_constant: float = GAS_CONSTANT) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.vapor_pressure_module`

Calculate vapor pressure using Clausius-Clapeyron equation.

This function calculates the final vapor pressure based on an initial
temperature/pressure pair and the latent heat of vaporization,
assuming ideal gas behavior.

- P_final = P_initial × exp( (L / R) × (1 / T_initial - 1 / T_final) )
    - P_final is Final vapor pressure [Pa].
    - P_initial is Initial vapor pressure [Pa].
    - L is Latent heat of vaporization [J/mol].
    - R is Universal gas constant [J/(mol·K)].
    - T_initial is Initial temperature [K].
    - T_final is Final temperature [K].

**Parameters**

- `latent_heat` (`Union[float, NDArray[np.float64]]`) — Latent heat of vaporization [J/mol].
- `temperature_initial` (`Union[float, NDArray[np.float64]]`) — Initial temperature [K].
- `pressure_initial` (`Union[float, NDArray[np.float64]]`) — Initial vapor pressure [Pa].
- `temperature` (`Union[float, NDArray[np.float64]]`) — Final temperature [K].
- `gas_constant` (`float`) (default: `GAS_CONSTANT`) — Gas constant (default 8.314 J/(mol·K)).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure [Pa].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_clausius_clapeyron_vapor_pressure(
    40660, 373.15, 101325, 300
)
# Output: ~35307 Pa
```

> **References**
>
> - https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/vapor_pressure_module.py#L64-L118)

## particula.gas.get_concentration_from_pressure

*function*

```python
def get_concentration_from_pressure(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.concentration_function`

Calculate the concentration of a gas using the ideal gas law.

The concentration is determined from the partial pressure, molar mass,
and temperature using the ideal gas equation:

- C = (P × M) / (R × T)
    - C is the concentration in kg/m³,
    - P is the partial pressure in Pascals (Pa),
    - M is the molar mass in kg/mol,
    - R is the universal gas constant (J/(mol·K)),
    - T is the temperature in Kelvin.

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas in Pascals (Pa).
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas in kg/mol.
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

**Examples**

```py title="Floating-point Example Usage"
import particula as par
par.gas.get_concentration_from_pressure(101325, 0.02897, 298.15)
# Output: 1.184587604735883
```

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Concentration of the gas in kg/m³.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/concentration_function.py#L11-L43)

## particula.gas.get_dynamic_viscosity

*function*

```python
def get_dynamic_viscosity(temperature: float, reference_viscosity: float = REF_VISCOSITY_AIR_STP, reference_temperature: float = REF_TEMPERATURE_STP) -> float
```

Re-exported from: `particula.gas.properties.dynamic_viscosity`

Calculate the dynamic viscosity of air using Sutherland's formula.

- μ(T) = μ₀ × (T / T₀)^(3/2) × (T₀ + S) / (T + S)
    - μ(T) is the dynamic viscosity at temperature T (Pa·s).
    - μ₀ is the reference viscosity (Pa·s).
    - T is the temperature in Kelvin.
    - T₀ is the reference temperature in Kelvin.
    - S is the Sutherland constant in Kelvin.

**Parameters**

- `temperature` (`float`) — Desired air temperature in Kelvin. Must be > 0.
- `reference_viscosity` (`float`) (default: `REF_VISCOSITY_AIR_STP`) — Gas viscosity at the reference temperature (default is STP).
- `reference_temperature` (`float`) (default: `REF_TEMPERATURE_STP`) — Gas temperature in Kelvin for the reference viscosity (default is STP).

**Returns**

- (`float`) — Dynamic viscosity of air at the given temperature in Pa·s.

**Examples**

``` py title="Example Float Usage"
import particula as par
par.gas.get_dynamic_viscosity(300.0)
# Output (approx.): 1.846e-05
```

> **References**
>
> - Wolfram Formula Repository, "Sutherland's Formula,"
>   https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/dynamic_viscosity.py#L27-L68)

## particula.gas.get_eulerian_integral_length

*function*

```python
def get_eulerian_integral_length(fluid_rms_velocity: Union[float, NDArray[np.float64]], turbulent_dissipation: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.integral_scale_module`

Calculate the Eulerian integral length scale.

The Eulerian integral length scale is a measure of the size of the largest
turbulent eddies in a fluid flow.

- L_e = 0.5 × (u'³) / ε
    - L_e is Eulerian integral length scale [m].
    - fluid_rms_velocity (u') is Fluid RMS fluctuation velocity [m/s].
    - turbulent_dissipation (ε) is Turbulent energy dissipation rate
        [m²/s³].

**Parameters**

- `fluid_rms_velocity` (`Union[float, NDArray[np.float64]]`) — Fluid RMS fluctuation velocity [m/s].
- `turbulent_dissipation` (`Union[float, NDArray[np.float64]]`) — Turbulent kinetic energy dissipation rate [m²/s³].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Eulerian integral length scale [m].

**Examples**

``` py title="Example"
import particula as par
par.gas.get_eulerian_integral_length(0.3, 1e-4)
# Output: 1350.0
```

> **References**
>
> - Hinze, J. O., "Turbulence," McGraw-Hill, 1975. [Check this reference]
> - Wikipedia contributors, "Turbulence," Wikipedia.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/integral_scale_module.py#L52-L89)

## particula.gas.get_fluid_rms_velocity

*function*

```python
def get_fluid_rms_velocity(re_lambda: Union[float, NDArray[np.float64]], kinematic_viscosity: Union[float, NDArray[np.float64]], turbulent_dissipation: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.fluid_rms_velocity`

Calculate the fluid RMS fluctuation velocity.

The fluid root-mean-square (RMS) velocity fluctuation quantifies
turbulence intensity in a fluid flow. It is calculated as:

- u' = (R_λ^(1/2) v_K) / 15^(1/4)
    - u' is Fluid RMS fluctuation velocity [m/s]
    - R_λ (re_lambda) is Taylor-microscale Reynolds number [-]
    - v_K is Kolmogorov velocity scale, computed as v_K = ( ε)^(1/4) [m/s]
    - v (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s]
    - ε (turbulent_dissipation) is Turbulent energy dissipation rate [m²/s³]

**Parameters**

- `re_lambda` (`Union[float, NDArray[np.float64]]`) — Taylor-microscale Reynolds number [-]
- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid [m²/s]
- `turbulent_dissipation` (`Union[float, NDArray[np.float64]]`) — Rate of dissipation of turbulent kinetic energy [m²/s³]

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Fluid RMS fluctuation velocity [m/s]

**Examples**

``` py title="Example Usage"
velocity = get_fluid_rms_velocity(500, 1.5e-5, 0.1)
# Output (example): 0.35
```

``` py title="Example Usage with Array Input"
velocity = get_fluid_rms_velocity(
    np.array([500, 600]),
    np.array([1.5e-5, 1.7e-5]),
    np.array([0.1, 0.12])
)
# Output (example): array([0.35, 0.41])
```

> **References**
>
> - H. Tennekes and J. L. Lumley, "A First Course in Turbulence,"
>   MIT Press, 1972. [check this]

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/fluid_rms_velocity.py#L12-L68)

## particula.gas.get_kinematic_viscosity

*function*

```python
def get_kinematic_viscosity(dynamic_viscosity: float, fluid_density: float) -> float
```

Re-exported from: `particula.gas.properties.kinematic_viscosity`

Calculate the kinematic viscosity of a fluid.

The function calculates ν by dividing the dynamic viscosity (μ)
by the fluid density (ρ).

- ν = μ / ρ
    - ν is Kinematic viscosity [m²/s].
    - μ is Dynamic viscosity [Pa·s].
    - ρ is Fluid density [kg/m³].

**Parameters**

- `dynamic_viscosity` (`float`) — Dynamic viscosity of the fluid [Pa·s].
- `fluid_density` (`float`) — Density of the fluid [kg/m³].

**Returns**

- (`float`) — The kinematic viscosity [m²/s].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_kinematic_viscosity(1.8e-5, 1.2)
# Output: ~1.5e-5
```

> **References**
>
> - "Viscosity Conversion Formula," Wolfram Formula Repository.
>   https://resources.wolframcloud.com/FormulaRepository/resources/Viscosity-Conversion-Formula

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kinematic_viscosity.py#L31-L64)

## particula.gas.get_kinematic_viscosity_via_system_state

*function*

```python
def get_kinematic_viscosity_via_system_state(temperature: float, fluid_density: float, reference_viscosity: float = REF_VISCOSITY_AIR_STP, reference_temperature: float = REF_TEMPERATURE_STP) -> float
```

Re-exported from: `particula.gas.properties.kinematic_viscosity`

Calculate the kinematic viscosity of air by first computing its dynamic
viscosity.

This function uses get_dynamic_viscosity(...) and divides by the given
fluid_density to get the kinematic viscosity.

- ν = μ / ρ
    - ν is Kinematic viscosity [m²/s].
    - μ is Dynamic viscosity [Pa·s].
    - ρ is Fluid density [kg/m³].

> **Where**
>
> - ν is Kinematic viscosity [m²/s].
> - μ is Dynamic viscosity [Pa·s].
> - ρ is Fluid density [kg/m³].

**Parameters**

- `temperature` (`float`) — Desired air temperature [K]. Must be > 0.
- `fluid_density` (`float`) — Density of the fluid [kg/m³].
- `reference_viscosity` (`float`) (default: `REF_VISCOSITY_AIR_STP`) — Reference dynamic viscosity [Pa·s].
- `reference_temperature` (`float`) (default: `REF_TEMPERATURE_STP`) — Reference temperature [K].

**Returns**

- (`float`) — The kinematic viscosity of air [m²/s].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_kinematic_viscosity_via_system_state(300, 1.2)
# Output: ~1.5e-5
```

> **References**
>
> - "Sutherland's Formula," Wolfram Formula Repository,
>   https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kinematic_viscosity.py#L67-L117)

## particula.gas.get_kolmogorov_length

*function*

```python
def get_kolmogorov_length(kinematic_viscosity: Union[float, NDArray[np.float64]], turbulent_dissipation: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.kolmogorov_module`

Calculate the Kolmogorov length scale.

The Kolmogorov length scale represents the smallest eddies in a turbulent
flow where viscosity dominates. It is defined as:

- η = (ν³ / ε)^(1/4)
    - η is the Kolmogorov length scale [m]
    - ν is the kinematic viscosity of the fluid [m^2/s]
    - ε is the rate of dissipation of turbulent kinetic energy [m^2/s^3]

> **Where**
>
> - η Kolmogorov length scale [m]
> - ν Kinematic viscosity of the fluid [m^2/s]
> - ε Rate of dissipation of turbulent kinetic energy [m^2/s^3]

**Parameters**

- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid [m^2/s]
- `turbulent_dissipation` (`Union[float, NDArray[np.float64]]`) — Rate of dissipation of turbulent kinetic energy [m^2/s^3]

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Kolmogorov length scale [m]

**Examples**

```py title="Kolmogorov length scale of a fluid"
import particula as par
par.gas.get_kolmogorov_length(1.5e-5, 0.1)
# Output: 0.0029154759474226504
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
> the geometric collision rate of sedimenting droplets. Part 2. Theory
> and parameterization. New Journal of Physics, 10.
> https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kolmogorov_module.py#L60-L103)

## particula.gas.get_kolmogorov_time

*function*

```python
def get_kolmogorov_time(kinematic_viscosity: Union[float, NDArray[np.float64]], turbulent_dissipation: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.kolmogorov_module`

Calculate the Kolmogorov time of a fluid.

The Kolmogorov time scale represents the smallest timescale in turbulence
where viscous forces dominate over inertial effects. This timescale
characterizes the turnover time of the smallest turbulent
eddies. It is given by:

- τ_K = (v / ε)^(1/2)
    - τ_K is the Kolmogorov time [s]
    - v is the kinematic viscosity of the fluid [m^2/s]

**Parameters**

- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid [m^2/s]
- `turbulent_dissipation` (`Union[float, NDArray[np.float64]]`) — Rate of dissipation of turbulent kinetic energy [m^2/s^3]

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Kolmogorov time [s]

**Examples**

```py title="Kolmogorov time of a fluid"
import particula as par
par.gas.get_kolmogorov_time(1.5e-5, 0.1)
# Output: 0.3872983346207417
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
> the geometric collision rate of sedimenting droplets. Part 2. Theory
> and parameterization. New Journal of Physics, 10.
> https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kolmogorov_module.py#L18-L57)

## particula.gas.get_kolmogorov_velocity

*function*

```python
def get_kolmogorov_velocity(kinematic_viscosity: Union[float, NDArray[np.float64]], turbulent_dissipation: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.kolmogorov_module`

Calculate the Kolmogorov velocity scale.

The Kolmogorov velocity scale characterizes the smallest turbulent velocity
fluctuations and is given by:

- v_k = (v ε)^(1/4)
    - v_k is the Kolmogorov velocity scale [m/s]
    - v is the kinematic viscosity of the fluid [m^2/s]
    - ε is the rate of dissipation of turbulent kinetic energy [m^2/s^3]

**Parameters**

- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid [m^2/s]
- `turbulent_dissipation` (`Union[float, NDArray[np.float64]]`) — Rate of dissipation of turbulent kinetic energy [m^2/s^3]

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Kolmogorov velocity scale [m/s]

**Examples**

```py title="Kolmogorov velocity scale of a fluid"
import particula as par
par.gas.get_kolmogorov_velocity(1.5e-5, 0.1)
# Output: 0.3872983346207417
```

> **References**
>
> - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
> the geometric collision rate of sedimenting droplets. Part 2. Theory
> and parameterization. New Journal of Physics, 10.
> https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/kolmogorov_module.py#L106-L144)

## particula.gas.get_lagrangian_integral_time

*function*

```python
def get_lagrangian_integral_time(fluid_rms_velocity: Union[float, NDArray[np.float64]], turbulent_dissipation: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.integral_scale_module`

Calculate the Lagrangian integral timescale.

The Lagrangian integral timescale is a measure of the time it takes for
a fluid particle to travel a distance equal to the integral length scale.

- T_L = (u'²) / ε
    - T_L is Lagrangian integral timescale [s].
    - fluid_rms_velocity (u') is Fluid RMS fluctuation velocity [m/s].
    - turbulent_dissipation (ε) is Turbulent energy dissipation rate
        [m²/s³].

**Parameters**

- `fluid_rms_velocity` (`Union[float, NDArray[np.float64]]`) — Fluid RMS fluctuation velocity [m/s].
- `turbulent_dissipation` (`Union[float, NDArray[np.float64]]`) — Turbulent kinetic energy dissipation rate [m²/s³].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Lagrangian integral timescale [s].

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_lagrangian_integral_time(0.3, 1e-4)
# Output: 900.0
```

> **References**
>
> - Townsend, A. A., "The Structure of Turbulent Shear Flow," 2nd ed.,
>   Cambridge University Press, 1976. [Check this reference]
> - Wikipedia contributors, "Turbulence," Wikipedia.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/integral_scale_module.py#L11-L49)

## particula.gas.get_lagrangian_taylor_microscale_time

*function*

```python
def get_lagrangian_taylor_microscale_time(kolmogorov_time: Union[float, NDArray[np.float64]], re_lambda: Union[float, NDArray[np.float64]], accel_variance: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.taylor_microscale_module`

Calculate the Lagrangian Taylor microscale time.

The Lagrangian Taylor microscale time (τ_T) represents the characteristic
time for the decay of turbulent velocity correlations. It provides insight
into the memory of turbulent fluid elements. It is given by:

- τ_T = τ_k * (2 R_λ / (15^(1/2) a_o))^(1/2)
    - τ_T is Lagrangian Taylor microscale time [s]
    - τ_k (kolmogorov_time) is Kolmogorov time scale [s]
    - R_λ (re_lambda) is Taylor-microscale Reynolds number [-]
    - a_o (accel_variance) is Normalized acceleration variance in isotropic
        turbulence [-]

**Parameters**

- `kolmogorov_time` (`Union[float, NDArray[np.float64]]`) — Kolmogorov time scale [s]
- `re_lambda` (`Union[float, NDArray[np.float64]]`) — Taylor-microscale Reynolds number [-]
- `accel_variance` (`Union[float, NDArray[np.float64]]`) — Normalized acceleration variance in isotropic turbulence [-]

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_lagrangian_taylor_microscale_time(0.387, 500, 0.05)
# Output: 0.3872983346207417
```

Returns:
    Lagrangian Taylor microscale time [s]

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L13-L55)

## particula.gas.get_molecule_mean_free_path

*function*

```python
def get_molecule_mean_free_path(molar_mass: Union[float, NDArray[np.float64]] = MOLECULAR_WEIGHT_AIR, temperature: float = 298.15, pressure: float = 101325, dynamic_viscosity: Optional[float] = None) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.mean_free_path`

Calculate the mean free path of a gas molecule in air.

This function calculates λ based on the input conditions. If
dynamic_viscosity is not provided, it is computed via
get_dynamic_viscosity(temperature).

- λ = (2 × μ / P) / √(8 × M / (π × R × T))
    - λ is Mean free path [m].
    - μ is Dynamic viscosity [Pa·s].
    - P is Gas pressure [Pa].
    - M is Molar mass [kg/mol].
    - R is Universal gas constant [J/(mol·K)].
    - T is Gas temperature [K].

**Parameters**

- `molar_mass` (`Union[float, NDArray[np.float64]]`) (default: `MOLECULAR_WEIGHT_AIR`) — The molar mass of the gas molecule [kg/mol].
- `temperature` (`float`) (default: `298.15`) — The temperature of the gas [K].
- `pressure` (`float`) (default: `101325`) — The pressure of the gas [Pa].
- `dynamic_viscosity` (`Optional[float]`) (default: `None`) — The dynamic viscosity of the gas [Pa·s]. If None, it will be calculated based on the temperature.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Mean free path of the gas molecule in meters (m).

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_molecule_mean_free_path()
# Returns mean free path at ~298K and 101325Pa, ~6.5e-8 m
```

> **References**
>
> - "Mean Free Path," Wikipedia, The Free Encyclopedia.
>   https://en.wikipedia.org/wiki/Mean_free_path

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/mean_free_path.py#L31-L88)

## particula.gas.get_normalized_accel_variance_ao2008

*function*

```python
def get_normalized_accel_variance_ao2008(re_lambda: Union[float, NDArray[np.float64]], numerical_stability_epsilon: float = 1e-14) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.normalize_accel_variance`

Calculate the normalized acceleration variance in isotropic turbulence.

This coefficient describes the statistical behavior of acceleration
fluctuations in turbulent flows.

- a_o = (11 + 7 R_λ) / (205 + R_λ)
    - a_o is Normalized acceleration variance in isotropic turbulence [-].
    - R_λ is Taylor-microscale Reynolds number [-].

> **Where**
>
> - a_o (accel_variance) is Normalized acceleration variance in isotropic
>   turbulence [-].
> - R_λ (re_lambda) is Taylor-microscale Reynolds number [-].
> - ε (numerical_stability_epsilon) is Small number added to R_λ
>   for numerical stability.

**Parameters**

- `re_lambda` (`Union[float, NDArray[np.float64]]`) — Taylor-microscale Reynolds number [-]

**Returns**

- `accel_variance` (`Union[float, NDArray[np.float64]]`) — Normalized acceleration variance [-]

**Examples**

```py title="Example Usage"
import particula as par
par.gas.get_normalized_accel_variance_ao2008(500.0)
# Output: ~0.05
```

References:
- The equivalent numerically stable version used is this.
    (7 + 11 / (R_λ + ε)) / (1 + 205 / (R_λ + ε))
- Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
    the geometric collision rate of sedimenting droplets. Part 2. Theory
    and parameterization. New Journal of Physics, 10.
    https://doi.org/10.1088/1367-2630/10/7/075016

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/normalize_accel_variance.py#L11-L55)

## particula.gas.get_partial_pressure

*function*

```python
def get_partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.pressure_function`

Calculate partial pressure from concentration, molar mass, T.

- p = (c × R × T) / M
    - p is Partial pressure [Pa].
    - c is Gas concentration [kg/m³].
    - R is Universal gas constant [J/(mol·K)].
    - T is Temperature [K].
    - M is Molar mass [kg/mol].

**Parameters**

- `concentration` (`Union[float, NDArray[np.float64]]`) — Concentration of the gas [kg/m³].
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass of the gas [kg/mol].
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature [K].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Partial pressure of the gas [Pa].

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_partial_pressure(1.2, 0.02897, 298)
# Output: ~986.4 Pa
```

> **References**
>
> - Wikipedia contributors, "Ideal gas law," Wikipedia,
>   https://en.wikipedia.org/wiki/Ideal_gas_law

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/pressure_function.py#L12-L53)

## particula.gas.get_saturation_ratio_from_pressure

*function*

```python
def get_saturation_ratio_from_pressure(partial_pressure: Union[float, NDArray[np.float64]], pure_vapor_pressure: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.pressure_function`

Calculate the saturation ratio of the gas at a given partial pressure and
pure vapor pressure.

The saturation ratio is defined as the ratio of partial pressure to the
pure vapor pressure.

- S = p / p_vap
    - S is Saturation ratio (dimensionless).
    - p is Partial pressure [Pa].
    - p_vap is Pure vapor pressure [Pa].

**Parameters**

- `partial_pressure` (`Union[float, NDArray[np.float64]]`) — Partial pressure [Pa].
- `pure_vapor_pressure` (`Union[float, NDArray[np.float64]]`) — Pure vapor pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Saturation ratio of the gas (dimensionless).

**Examples**

```py title="Example usage"
import particula as par
par.gas.get_saturation_ratio_from_pressure(800.0, 1000.0)
# Output: 0.8
```

> **References**
>
> - Wikipedia contributors, "Relative humidity," Wikipedia,
>   https://en.wikipedia.org/wiki/Relative_humidity

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/pressure_function.py#L56-L95)

## particula.gas.get_taylor_microscale

*function*

```python
def get_taylor_microscale(fluid_rms_velocity: Union[float, NDArray[np.float64]], kinematic_viscosity: Union[float, NDArray[np.float64]], turbulent_dissipation: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.taylor_microscale_module`

Calculate the Taylor microscale.

The Taylor microscale (λ) represents an intermediate length scale in
turbulence, linking the dissipative and energy-containing ranges of
turbulence. It characterizes the smoothness of velocity fluctuations
in turbulent flows. It is given by:

- λ = u' * (15 ν² / ε)^(1/2)
    - λ is Taylor microscale [m]
    - u' (rms_velocity) is Fluid RMS fluctuation velocity [m/s]
    - v (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s]
    - ε (turbulent_dissipation) is Turbulent kinetic energy dissipation
        rate [m²/s³]

**Parameters**

- `fluid_rms_velocity` (`Union[float, NDArray[np.float64]]`) — Fluid RMS fluctuation velocity [m/s]
- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid [m²/s]
- `turbulent_dissipation` (`Union[float, NDArray[np.float64]]`) — Turbulent kinetic energy dissipation rate [m²/s³]

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Taylor microscale [m]

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_taylor_microscale(0.35, 1.5e-5, 0.1)
# Output: 0.00021081851067789195
```

> **References**
>
> - https://en.wikipedia.org/wiki/Taylor_microscale

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L58-L105)

## particula.gas.get_taylor_microscale_reynolds_number

*function*

```python
def get_taylor_microscale_reynolds_number(fluid_rms_velocity: Union[float, NDArray[np.float64]], taylor_microscale: Union[float, NDArray[np.float64]], kinematic_viscosity: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.taylor_microscale_module`

Compute the Taylor-microscale Reynolds number (Re_λ).

The Taylor-scale micro Reynolds number is a dimensionless quantity used in
turbulence studies to characterize the relative importance of inertial and
viscous forces at the Taylor microscale.

- Re_λ = (u' λ) / ν
    - u' (fluid_rms_velocity) is Fluid (RMS) velocity fluctuation [m/s].
    - λ (taylor_microscale) is Taylor microscale [m].
    - ν (kinematic_viscosity) is Kinematic viscosity of the fluid [m²/s].

**Parameters**

- `fluid_rms_velocity` (`Union[float, NDArray[np.float64]]`) — Fluid RMS velocity fluctuation [m/s].
- `taylor_microscale` (`Union[float, NDArray[np.float64]]`) — Taylor microscale [m].
- `kinematic_viscosity` (`Union[float, NDArray[np.float64]]`) — Kinematic viscosity of the fluid [m²/s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Taylor-microscale Reynolds number [dimensionless].

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_taylor_microscale_reynolds_number(0.35, 0.00021, 1.5e-5)
# Output: 500.0
```

> **References**
>
> - https://en.wikipedia.org/wiki/Taylor_microscale

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/taylor_microscale_module.py#L108-L149)

## particula.gas.get_thermal_conductivity

*function*

```python
def get_thermal_conductivity(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Re-exported from: `particula.gas.properties.thermal_conductivity`

Thermal conductivity of air as a function of temperature.

Calculate the thermal conductivity of air as a function of temperature.
Based on a simplified linear relation from atmospheric science literature.
Only valid for temperatures within the range typically found on
Earth's surface.

- k(T) = 1e-3 × (4.39 + 0.071 × T)
    - k(T) is Thermal conductivity [W/(m·K)].
    - T is Temperature [K].

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — The temperature in Kelvin (K).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The thermal conductivity [W/(m·K)] or [J/(m·s·K)].

**Examples**

``` py title="Example Usage"
import particula as par
par.gas.get_thermal_conductivity(300)
# ~0.449 W/(m·K)
```

References:
- Seinfeld and Pandis, "Atmospheric Chemistry and Physics", Equation 17.54.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/thermal_conductivity.py#L14-L45)

**Modules**

- `particula.gas.environment_data`
- `particula.gas.gas_data`
- `particula.gas.gas_data_builder`
- `particula.gas.latent_heat_builders`
- `particula.gas.latent_heat_factories`
- `particula.gas.latent_heat_strategies`
- `particula.gas.properties`
- `particula.gas.vapor_pressure_builders`
- `particula.gas.vapor_pressure_factories`
- `particula.gas.vapor_pressure_strategies`
