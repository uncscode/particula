# particula.particles.properties.kelvin_effect_module

*module*

Module to calculate the Kelvin effect on vapor pressure.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kelvin_effect_module.py#L1-L1)

## particula.particles.properties.kelvin_effect_module.MAX_KELVIN_RATIO

*attribute* · *module attribute*

```python
MAX_KELVIN_RATIO = 100.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kelvin_effect_module.py#L15-L15)

## particula.particles.properties.kelvin_effect_module.get_kelvin_radius

*function*

```python
def get_kelvin_radius(effective_surface_tension: Union[float, NDArray[np.float64]], effective_density: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], temperature: float) -> Union[float, NDArray[np.float64]]
```

Compute the Kelvin radius (rₖ) to account for curvature effects on vapor
pressure.

The Kelvin radius is defined by:

- rₖ = (2 × σ × M) / (R × T × ρ)
    - rₖ is Kelvin radius in meters (m).
    - σ is the effective surface tension in N/m.
    - M is the molar mass in kg/mol.
    - R is the universal gas constant in J/(mol·K).
    - T is the temperature in Kelvin (K).
    - ρ is the effective density in kg/m³.

**Parameters**

- `effective_surface_tension` (`Union[float, NDArray[np.float64]]`) — Surface tension of the mixture (N/m).
- `effective_density` (`Union[float, NDArray[np.float64]]`) — Effective density of the mixture (kg/m³).
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass (kg/mol).
- `temperature` (`float`) — Temperature of the system (K).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Kelvin radius in meters (float or NDArray[np.float64]).

**Examples**

``` py title="Example"
import numpy as np
import particula as par
par.particles.get_kelvin_radius(
    effective_surface_tension=0.072,
    effective_density=1000.0,
    molar_mass=0.018,
    temperature=298.15
)
# Output: ...
```

> **References**
>
> - "Kelvin equation," Wikipedia,
>   https://en.wikipedia.org/wiki/Kelvin_equation

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kelvin_effect_module.py#L18-L73)

## particula.particles.properties.kelvin_effect_module.get_kelvin_term

*function*

```python
def get_kelvin_term(particle_radius: Union[float, NDArray[np.float64]], kelvin_radius_value: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Compute the Kelvin exponential term to account for curvature effects.

The Kelvin term (K) is given by:

- K = exp(rₖ / rₚ)
    - K is dimensionless.
    - rₖ is the Kelvin radius (m).
    - rₚ is the particle radius (m).

For very small particles (< 1 nm), the ratio rₖ / rₚ can become
extremely large, leading to numerical overflow. To prevent this, the
ratio is clipped to a maximum value of 100, corresponding to a Kelvin
term of ~2.7e43. This is physically unrealistic but ensures numerical
stability. Below ~0.1 nm, continuum mechanics breaks down anyway, so
the condensation equations become questionable.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Radius of the particle (m).
- `kelvin_radius_value` (`Union[float, NDArray[np.float64]]`) — Precomputed Kelvin radius (m).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Dimensionless exponential factor adjusting vapor pressure. For extreme cases, the value is clipped to prevent overflow.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_kelvin_term(
    particle_radius=1e-7,
    kelvin_radius_value=2e-7
)
print(kv_term)
# Output: ...
```

> **References**
>
> - Donahue, N. M., et al. (2013). "How do organic vapors contribute to
>   new-particle formation?" Faraday Discussions, 165, 91–104.
>   https://doi.org/10.1039/C3FD00046J. [check]

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/kelvin_effect_module.py#L76-L153)
