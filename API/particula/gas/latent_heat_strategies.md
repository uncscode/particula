# particula.gas.latent_heat_strategies

*module*

Latent heat strategies for gas species.

Latent heat values are expressed in J/kg for per-species mass-based
calculations. Strategies return scalars for scalar temperature inputs and
arrays matching the input shape for array temperature inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L1-L1)

## particula.gas.latent_heat_strategies.ConstantLatentHeat

*class*

```python
class ConstantLatentHeat(LatentHeatStrategy)
```

Bases: `particula.gas.latent_heat_strategies.LatentHeatStrategy`

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

### particula.gas.latent_heat_strategies.ConstantLatentHeat.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float = latent_heat_ref
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L62-L62)

### particula.gas.latent_heat_strategies.ConstantLatentHeat.__init__

*method*

```python
def __init__(latent_heat_ref: float) -> None
```

Initialize the constant latent heat strategy.

**Parameters**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L55-L62)

### particula.gas.latent_heat_strategies.ConstantLatentHeat.latent_heat

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

## particula.gas.latent_heat_strategies.LatentHeatStrategy

*class*

```python
class LatentHeatStrategy(ABC)
```

Bases: `ABC`

Abstract base class for latent heat calculations.

Subclasses implement :meth:`latent_heat` to return the latent heat of
vaporization in J/kg for scalar or array temperature inputs.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L16-L34)

### particula.gas.latent_heat_strategies.LatentHeatStrategy.latent_heat

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

## particula.gas.latent_heat_strategies.LinearLatentHeat

*class*

```python
class LinearLatentHeat(LatentHeatStrategy)
```

Bases: `particula.gas.latent_heat_strategies.LatentHeatStrategy`

Latent heat strategy using a linear temperature dependence.

Computes a linear relation for each temperature:
:math:`L(T) = L_{ref} - slope * (T - T_{ref})`.

**Attributes**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg at the reference temperature.
- `slope` (`float`) — Linear temperature slope in J/(kg·K).
- `temperature_ref` (`float`) — Reference temperature in Kelvin.

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L87-L146)

### particula.gas.latent_heat_strategies.LinearLatentHeat.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float = latent_heat_ref
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L120-L120)

### particula.gas.latent_heat_strategies.LinearLatentHeat.slope

*attribute* · *instance attribute*

```python
slope: float = slope
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L121-L121)

### particula.gas.latent_heat_strategies.LinearLatentHeat.temperature_ref

*attribute* · *instance attribute*

```python
temperature_ref: float = temperature_ref
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L122-L122)

### particula.gas.latent_heat_strategies.LinearLatentHeat.__init__

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

### particula.gas.latent_heat_strategies.LinearLatentHeat.latent_heat

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

## particula.gas.latent_heat_strategies.PowerLawLatentHeat

*class*

```python
class PowerLawLatentHeat(LatentHeatStrategy)
```

Bases: `particula.gas.latent_heat_strategies.LatentHeatStrategy`

Latent heat strategy using a power-law temperature dependence.

Computes a power-law relation:
:math:`L(T) = L_{ref} * (1 - T / T_c)^{beta}`.
The ratio is clipped to [0, 1] to enforce non-negative latent heat values.

**Attributes**

- `latent_heat_ref` (`float`) — Reference latent heat in J/kg.
- `critical_temperature` (`float`) — Critical temperature in Kelvin.
- `beta` (`float`) — Power-law exponent (dimensionless).

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L149-L219)

### particula.gas.latent_heat_strategies.PowerLawLatentHeat.beta

*attribute* · *instance attribute*

```python
beta: float = beta
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L184-L184)

### particula.gas.latent_heat_strategies.PowerLawLatentHeat.critical_temperature

*attribute* · *instance attribute*

```python
critical_temperature: float = critical_temperature
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L183-L183)

### particula.gas.latent_heat_strategies.PowerLawLatentHeat.latent_heat_ref

*attribute* · *instance attribute*

```python
latent_heat_ref: float = latent_heat_ref
```

[View source](https://github.com/uncscode/particula/blob/main/particula/gas/latent_heat_strategies.py#L182-L182)

### particula.gas.latent_heat_strategies.PowerLawLatentHeat.__init__

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

### particula.gas.latent_heat_strategies.PowerLawLatentHeat.latent_heat

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
