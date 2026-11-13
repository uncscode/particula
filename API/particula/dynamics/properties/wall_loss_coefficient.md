# particula.dynamics.properties.wall_loss_coefficient

*module*

Wall loss coefficient calculations for particles in chambers.

This module provides functions to calculate the wall loss coefficients
for particles in a chamber, either spherical or rectangular. These
coefficients are crucial in understanding particle lifetimes, as they
define the rates at which particles deposit onto chamber walls under
various conditions (e.g., eddy diffusivity, settling velocity, diffusion
coefficient).

> **References**
>
> - Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
>   GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
>   SHAPE. In J Aerosol Sct (Vol. 12, Issue 5).
>   https://doi.org/10.1016/0021-8502(81)90036-7
> - Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
>   loss rates in vessels. Aerosol Science and Technology, 2(3), 303-309.
>   https://doi.org/10.1080/02786828308958636
> - McMurry, P. H., & Rader, D. J. (1985). Aerosol Wall Losses in
>   Electrically Charged Chambers. Aerosol Science and Technology, 4(3),
>   249-268. https://doi.org/10.1080/02786828508959054

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L1-L1)

## particula.dynamics.properties.wall_loss_coefficient.get_rectangle_wall_loss_coefficient

*function*

```python
def get_rectangle_wall_loss_coefficient(wall_eddy_diffusivity: Union[float, NDArray[np.float64]], diffusion_coefficient: Union[float, NDArray[np.float64]], settling_velocity: Union[float, NDArray[np.float64]], chamber_dimensions: Tuple[float, float, float]) -> Union[float, NDArray[np.float64]]
```

Calculate the particle wall loss coefficient in a rectangular chamber.

This function computes the wall loss coefficient (β₀) for a rectangular
chamber of length (L), width (W), and height (H). It uses the wall eddy
diffusivity, particle diffusion coefficient, particle settling velocity,
and chamber dimensions:

- β₀ ~ (some function of wall_eddy_diffusivity, diffusion_coefficient,
settling_velocity, and L×W×H)

**Parameters**

- `wall_eddy_diffusivity` (`Union[float, NDArray[np.float64]]`) — Wall eddy diffusivity in s⁻¹.
- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) — Particle diffusion coefficient in m²/s.
- `settling_velocity` (`Union[float, NDArray[np.float64]]`) — Particle settling velocity in m/s.
- `chamber_dimensions` (`Tuple[float, float, float]`) — A tuple (length, width, height) in m.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The wall loss coefficient β₀ (float or NDArray[np.float64]), in s⁻¹.

Examples:
```py title="Example (float inputs)"
from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient
)

beta_0 = get_rectangle_wall_loss_coefficient(
    wall_eddy_diffusivity=1e-3,
    diffusion_coefficient=1e-5,
    settling_velocity=2e-4,
    chamber_dimensions=(1.0, 0.5, 0.5)
)
print(beta_0)
# Example output: 0.0009
```

References:
- Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
  GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
  SHAPE. J Aerosol Sci, 12(5).
  https://doi.org/10.1016/0021-8502(81)90036-7

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L100-L163)

## particula.dynamics.properties.wall_loss_coefficient.get_rectangle_wall_loss_coefficient_via_system_state

*function*

```python
def get_rectangle_wall_loss_coefficient_via_system_state(wall_eddy_diffusivity: float, particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], temperature: float, pressure: float, chamber_dimensions: Tuple[float, float, float]) -> Union[float, NDArray[np.float64]]
```

Calculate the wall loss coefficient for a rectangular chamber based on
the system state.

This function computes the wall loss coefficient for a rectangular chamber
using the system's physical state, including the wall eddy diffusivity,
particle properties (radius, density), and environmental conditions
(temperature, pressure). The chamber dimensions (length, width, height)
are also considered.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — The rate of wall eddy diffusivity in inverse seconds (s⁻¹).
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle in meters (m).
- `particle_density` (`Union[float, NDArray[np.float64]]`) — The density of the particle in kilograms per cubic meter (kg/m³).
- `temperature` (`float`) — The temperature of the system in Kelvin (K).
- `pressure` (`float`) — The pressure of the system in Pascals (Pa).
- `chamber_dimensions` (`Tuple[float, float, float]`) — A tuple containing the length, width, and height of the rectangular chamber in meters (m).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The calculated wall loss coefficient for the rectangular chamber.

> **References**
>
> - Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
>     GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
>     SHAPE. In J Aerosol Sct (Vol. 12, Issue 5).
>     https://doi.org/10.1016/0021-8502(81)90036-7

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L248-L306)

## particula.dynamics.properties.wall_loss_coefficient.get_spherical_wall_loss_coefficient

*function*

```python
def get_spherical_wall_loss_coefficient(wall_eddy_diffusivity: Union[float, NDArray[np.float64]], diffusion_coefficient: Union[float, NDArray[np.float64]], settling_velocity: Union[float, NDArray[np.float64]], chamber_radius: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the particle wall loss coefficient in a spherical chamber.

This function computes the wall loss coefficient based on a spherical
chamber approximation. It uses the wall eddy diffusivity, particle
diffusion coefficient, particle settling velocity, and chamber radius.
The calculation is:

- k = 6 × √(Dₑ × D) / (π × R) × f + vₛ × (3 / (4 × R))
    - k is the wall loss coefficient [s⁻¹],
    - Dₑ is the wall eddy diffusivity [m²/s or effective rate],
    - D is the particle diffusion coefficient [m²/s],
    - f is the Debye function evaluation (unitless),
    - vₛ is the settling velocity [m/s],
    - R is the chamber radius [m].

**Parameters**

- `wall_eddy_diffusivity` (`Union[float, NDArray[np.float64]]`) — The wall eddy diffusivity (or rate) in s⁻¹.
- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) — The diffusion coefficient of the particle in m²/s.
- `settling_velocity` (`Union[float, NDArray[np.float64]]`) — The particle settling velocity in m/s.
- `chamber_radius` (`Union[float, NDArray[np.float64]]`) — The spherical chamber radius in m.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The wall loss coefficient k, in inverse seconds (float or NDArray[np.float64]).

Examples:
```py title="Example (float inputs)"
from particula.dynamics.properties.wall_loss_coefficient import (
    get_spherical_wall_loss_coefficient
)

k_value = get_spherical_wall_loss_coefficient(
    wall_eddy_diffusivity=1e-2,
    diffusion_coefficient=5e-6,
    settling_velocity=1e-4,
    chamber_radius=0.5
)
print(k_value)
# Example output: 0.0012
```

References:
- Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
  loss rates in vessels. Aerosol Science and Technology, 2(3), 303-309.
  https://doi.org/10.1080/02786828308958636

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L35-L97)

## particula.dynamics.properties.wall_loss_coefficient.get_spherical_wall_loss_coefficient_via_system_state

*function*

```python
def get_spherical_wall_loss_coefficient_via_system_state(wall_eddy_diffusivity: float, particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], temperature: float, pressure: float, chamber_radius: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate spherical wall loss coefficient via system state.

Uses the system's physical conditions (particle radius, density,
temperature, pressure) to compute the needed diffusion and settling velocity
before calculating the spherical wall loss coefficient:

- k = f(
    wall_eddy_diffusivity,
    diffusion_coefficient_via_system_state,
    settling_velocity_via_system_state,
    chamber_radius
)

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity in s⁻¹.
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in m.
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density in kg/m³.
- `temperature` (`float`) — System temperature in K.
- `pressure` (`float`) — System pressure in Pa.
- `chamber_radius` (`Union[float, NDArray[np.float64]]`) — Chamber radius in m.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The computed wall loss coefficient k (float or NDArray[np.float64]) in s⁻¹.

Examples:
```py title="Example"
from particula.dynamics.properties.wall_loss_coefficient import (
    get_spherical_wall_loss_coefficient_via_system_state
)

k_value = get_spherical_wall_loss_coefficient_via_system_state(
    wall_eddy_diffusivity=1e-2,
    particle_radius=1e-7,
    particle_density=1000,
    temperature=298,
    pressure=101325,
    chamber_radius=0.5
)
print(k_value)
# Example output: 0.0018
```

References:
- Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
  loss rates in vessels. Aerosol Science and Technology, 2(3), 303-309.
  https://doi.org/10.1080/02786828308958636

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/properties/wall_loss_coefficient.py#L167-L244)
