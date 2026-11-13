# particula.particles.properties.diffusion_coefficient

*module*

Particle diffusion coefficient calculation.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusion_coefficient.py#L1-L1)

## particula.particles.properties.diffusion_coefficient.get_diffusion_coefficient

*function*

```python
def get_diffusion_coefficient(temperature: Union[float, NDArray[np.float64]], aerodynamic_mobility: Union[float, NDArray[np.float64]], boltzmann_constant: float = BOLTZMANN_CONSTANT) -> Union[float, NDArray[np.float64]]
```

Calculate the diffusion coefficient of a particle based on temperature
and aerodynamic mobility.

The diffusion coefficient (D) can be computed using:

- D = k_B T × B
    - D is the diffusion coefficient in m²/s,
    - k_B is the Boltzmann constant in J/K,
    - T is the temperature in Kelvin,
    - B is the aerodynamic mobility in m²/s.

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature in Kelvin (K).
- `aerodynamic_mobility` (`Union[float, NDArray[np.float64]]`) — Aerodynamic mobility in m²/s.
- `boltzmann_constant` (`float`) (default: `BOLTZMANN_CONSTANT`) — Boltzmann constant in J/K.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The diffusion coefficient of the particle in m²/s.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_diffusion_coefficient(
    temperature=300.0, aerodynamic_mobility=1.0e-8
)
# Output: ...
```

> **References**
>
> - Einstein, A. (1905). "On the movement of small particles suspended
>   in stationary liquids required by the molecular-kinetic theory of
>   heat." Annalen der Physik, 17(8), 549–560.
> - "Stokes-Einstein equation," Wikipedia,
>   https://en.wikipedia.org/wiki/Stokes%E2%80%93Einstein_equation

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusion_coefficient.py#L23-L69)

## particula.particles.properties.diffusion_coefficient.get_diffusion_coefficient_via_system_state

*function*

```python
def get_diffusion_coefficient_via_system_state(particle_radius: Union[float, NDArray[np.float64]], temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Calculate the diffusion coefficient from system state parameters.

This function determines the diffusion coefficient (D) of a particle by:
1. Computing gas properties (dynamic viscosity, mean free path),
2. Determining particle slip correction and aerodynamic mobility,
3. Calling get_diffusion_coefficient() to get D.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters (m).
- `temperature` (`float`) — System temperature in Kelvin (K).
- `pressure` (`float`) — System pressure in Pascals (Pa).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The diffusion coefficient of the particle in m²/s.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_diffusion_coefficient_via_system_state(
    particle_radius=1.0e-7,
    temperature=298.15,
    pressure=101325
)
# Output: ...
```

> **References**
>
> - Millikan, R. A. (1923). "On the elementary electrical charge and the
>   Avogadro constant." Physical Review, 2(2), 109–143. [check]
> - "Mass Diffusion," Wikipedia,
>   https://en.wikipedia.org/wiki/Diffusion#Mass_diffusion

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/diffusion_coefficient.py#L72-L134)
