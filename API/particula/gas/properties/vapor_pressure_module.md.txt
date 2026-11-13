# particula.gas.properties.vapor_pressure_module

*module*

Vapor pressure modules for calculating the vapor pressure of a gas.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/properties/vapor_pressure_module.py#L1-L1)

## particula.gas.properties.vapor_pressure_module.get_antoine_vapor_pressure

*function*

```python
def get_antoine_vapor_pressure(a: Union[float, NDArray[np.float64]], b: Union[float, NDArray[np.float64]], c: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

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

## particula.gas.properties.vapor_pressure_module.get_buck_vapor_pressure

*function*

```python
def get_buck_vapor_pressure(temperature: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

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

## particula.gas.properties.vapor_pressure_module.get_clausius_clapeyron_vapor_pressure

*function*

```python
def get_clausius_clapeyron_vapor_pressure(latent_heat: Union[float, NDArray[np.float64]], temperature_initial: Union[float, NDArray[np.float64]], pressure_initial: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]], gas_constant: float = GAS_CONSTANT) -> Union[float, NDArray[np.float64]]
```

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
