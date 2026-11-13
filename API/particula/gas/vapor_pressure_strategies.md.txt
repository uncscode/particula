# particula.gas.vapor_pressure_strategies

*module*

Vapor Pressure Module.

This module calculates the vapor pressure of substances based on different
strategies. These strategies are interchangeable and can be used to calculate
the vapor pressure of a substance at a given temperature.

All units are in base SI units (kg, m, s).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L1-L1)

## particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy

*class*

```python
class AntoineVaporPressureStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

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

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.MMHG_TO_PA

*attribute* · *class attribute* · *instance attribute*

```python
MMHG_TO_PA = 133.322
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L282-L282)

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.a

*attribute* · *instance attribute*

```python
a = a
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L297-L297)

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.b

*attribute* · *instance attribute*

```python
b = b
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L298-L298)

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.c

*attribute* · *instance attribute*

```python
c = c
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L299-L299)

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.__init__

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

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.pure_vapor_pressure

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

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.concentration

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

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.partial_pressure

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

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.saturation_concentration

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

### particula.gas.vapor_pressure_strategies.AntoineVaporPressureStrategy.saturation_ratio

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

## particula.gas.vapor_pressure_strategies.ArblasterLiquidVaporPressureStrategy

*class*

```python
class ArblasterLiquidVaporPressureStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

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

### particula.gas.vapor_pressure_strategies.ArblasterLiquidVaporPressureStrategy.coefficients

*attribute* · *instance attribute*

```python
coefficients = coefficients
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L571-L571)

### particula.gas.vapor_pressure_strategies.ArblasterLiquidVaporPressureStrategy.__init__

*method*

```python
def __init__(coefficients: tuple[float, float, float, float, float])
```

Initialize with the 5-term polynomial coefficients.

**Parameters**

- `coefficients` (`tuple[float, float, float, float, float]`) — Tuple ``(A, B, C, D, E)`` for the correlation.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L565-L571)

### particula.gas.vapor_pressure_strategies.ArblasterLiquidVaporPressureStrategy.pure_vapor_pressure

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

### particula.gas.vapor_pressure_strategies.ArblasterLiquidVaporPressureStrategy.concentration

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

### particula.gas.vapor_pressure_strategies.ArblasterLiquidVaporPressureStrategy.partial_pressure

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

### particula.gas.vapor_pressure_strategies.ArblasterLiquidVaporPressureStrategy.saturation_concentration

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

### particula.gas.vapor_pressure_strategies.ArblasterLiquidVaporPressureStrategy.saturation_ratio

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

## particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy

*class*

```python
class ClausiusClapeyronStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

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

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.latent_heat

*attribute* · *instance attribute*

```python
latent_heat = latent_heat
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L380-L380)

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.pressure_initial

*attribute* · *instance attribute*

```python
pressure_initial = pressure_initial
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L382-L382)

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.temperature_initial

*attribute* · *instance attribute*

```python
temperature_initial = temperature_initial
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L381-L381)

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.__init__

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

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.pure_vapor_pressure

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

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.concentration

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

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.partial_pressure

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

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.saturation_concentration

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

### particula.gas.vapor_pressure_strategies.ClausiusClapeyronStrategy.saturation_ratio

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

## particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy

*class*

```python
class ConstantVaporPressureStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

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

### particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy.vapor_pressure

*attribute* · *instance attribute*

```python
vapor_pressure = vapor_pressure
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L222-L222)

### particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy.__init__

*method*

```python
def __init__(vapor_pressure: Union[float, NDArray[np.float64]])
```

Initialize the Constant Vapor Pressure strategy.

**Parameters**

- `vapor_pressure` (`Union[float, NDArray[np.float64]]`) — The constant vapor pressure value in Pascals.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L216-L222)

### particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy.pure_vapor_pressure

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

### particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy.concentration

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

### particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy.partial_pressure

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

### particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy.saturation_concentration

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

### particula.gas.vapor_pressure_strategies.ConstantVaporPressureStrategy.saturation_ratio

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

## particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy

*class*

```python
class LiquidClausiusHybridStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

Hybrid strategy blending liquid and Clausius–Clapeyron methods.

This strategy uses the 5-term liquid polynomial at low temperature and
gradually transitions to the Clausius–Clapeyron relation above the
boiling point. A logistic weight provides a smooth changeover.

Methods:
- pure_vapor_pressure: Compute the blended vapor pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L600-L661)

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.boiling_point

*attribute* · *instance attribute*

```python
boiling_point = boiling_point
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L636-L636)

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.clausius_strategy

*attribute* · *instance attribute*

```python
clausius_strategy = ClausiusClapeyronStrategy(latent_heat, temperature_initial, pressure_initial)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L633-L635)

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.liquid_strategy

*attribute* · *instance attribute*

```python
liquid_strategy = ArblasterLiquidVaporPressureStrategy(coefficients)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L630-L632)

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.transition_width

*attribute* · *instance attribute*

```python
transition_width = transition_width
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L637-L637)

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.__init__

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

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.pure_vapor_pressure

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

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.concentration

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

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.partial_pressure

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

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.saturation_concentration

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

### particula.gas.vapor_pressure_strategies.LiquidClausiusHybridStrategy.saturation_ratio

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

## particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy

*class*

```python
class TableVaporPressureStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

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

### particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy.temperatures

*attribute* · *instance attribute*

```python
temperatures = table_temps[idx]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L528-L528)

### particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy.vapor_pressures

*attribute* · *instance attribute*

```python
vapor_pressures = table_pressures[idx]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L527-L527)

### particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy.__init__

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

### particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy.pure_vapor_pressure

*method*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Interpolate the vapor pressure for the given temperature.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L530-L540)

### particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy.concentration

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

### particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy.partial_pressure

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

### particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy.saturation_concentration

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

### particula.gas.vapor_pressure_strategies.TableVaporPressureStrategy.saturation_ratio

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

## particula.gas.vapor_pressure_strategies.VaporPressureStrategy

*class*

```python
class VaporPressureStrategy(ABC)
```

Bases: `ABC`

Base class for vapor pressure calculations.

This abstract class defines standard methods for partial pressure,
concentration, saturation ratio, saturation concentration, and pure vapor
pressure. Subclasses must implement the pure_vapor_pressure method
with specific formulae or empirical correlations for vapor pressure.

Methods:
- partial_pressure: Compute partial pressure from concentration.
- concentration: Compute concentration from partial pressure.
- saturation_ratio: Compute ratio of partial pressure to saturation
  pressure.
- saturation_concentration: Compute concentration at saturation pressure.
- pure_vapor_pressure: Abstract method to compute pure (saturation) vapor
  pressure.

**Examples**

```py title="General Usage"
# Cannot instantiate directly:
#    strategy = VaporPressureStrategy()  # Error (abstract)
# Use a derived strategy class instead.
```

> **References**
>
> - "Vapor Pressure,"
> [Wikipedia](https://en.wikipedia.org/wiki/Vapor_pressure).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L31-L183)

### particula.gas.vapor_pressure_strategies.VaporPressureStrategy.concentration

*method*

```python
def concentration(partial_pressure: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

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

### particula.gas.vapor_pressure_strategies.VaporPressureStrategy.partial_pressure

*method*

```python
def partial_pressure(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

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

### particula.gas.vapor_pressure_strategies.VaporPressureStrategy.pure_vapor_pressure

*method* · *abstract*

```python
def pure_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the pure (saturation) vapor pressure at a given
temperature. Units are in Pascals Pa=kg/(m·s²).

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/vapor_pressure_strategies.py#L174-L183)

### particula.gas.vapor_pressure_strategies.VaporPressureStrategy.saturation_concentration

*method*

```python
def saturation_concentration(molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

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

### particula.gas.vapor_pressure_strategies.VaporPressureStrategy.saturation_ratio

*method*

```python
def saturation_ratio(concentration: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

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

## particula.gas.vapor_pressure_strategies.WaterBuckStrategy

*class*

```python
class WaterBuckStrategy(VaporPressureStrategy)
```

Bases: `particula.gas.vapor_pressure_strategies.VaporPressureStrategy`

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

### particula.gas.vapor_pressure_strategies.WaterBuckStrategy.pure_vapor_pressure

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

### particula.gas.vapor_pressure_strategies.WaterBuckStrategy.concentration

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

### particula.gas.vapor_pressure_strategies.WaterBuckStrategy.partial_pressure

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

### particula.gas.vapor_pressure_strategies.WaterBuckStrategy.saturation_concentration

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

### particula.gas.vapor_pressure_strategies.WaterBuckStrategy.saturation_ratio

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
