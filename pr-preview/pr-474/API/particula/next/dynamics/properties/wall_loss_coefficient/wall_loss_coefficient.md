# Wall Loss Coefficient

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Properties](./index.md#properties) / Wall Loss Coefficient

> Auto-generated documentation for [particula.next.dynamics.properties.wall_loss_coefficient](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/properties/wall_loss_coefficient.py) module.

## rectangle_wall_loss_coefficient

[Show source in wall_loss_coefficient.py:65](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/properties/wall_loss_coefficient.py#L65)

Calculate the wall loss coefficient, β₀, for a rectangular chamber.

This function computes the wall loss coefficient for a rectangular-prism
chamber, considering the wall eddy diffusivity, particle diffusion
coefficient, and terminal settling velocity. The chamber dimensions
(length, width, and height) are used to account for the geometry's impact
on particle loss.

#### Arguments

- `wall_eddy_diffusivity` - Rate of wall diffusivity parameter in
    units of inverse seconds (s^-1).
- `diffusion_coefficient` - The particle diffusion coefficient
    in units of square meters per second (m²/s).
- `settling_velocity` - The terminal settling velocity of the
    particles, in units of meters per second (m/s).
- `chamber_dimensions` - A tuple of three floats representing the length
    (L), width (W), and height (H) of the rectangular chamber,
    in units of meters (m).

#### Returns

The calculated wall loss rate (β₀) for the rectangular chamber.

#### References

- Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
    GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
    SHAPE. In J Aerosol Sct (Vol. 12, Issue 5).
    https://doi.org/10.1016/0021-8502(81)90036-7

#### Signature

```python
def rectangle_wall_loss_coefficient(
    wall_eddy_diffusivity: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]],
    settling_velocity: Union[float, NDArray[np.float64]],
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]: ...
```



## rectangle_wall_loss_coefficient_via_system_state

[Show source in wall_loss_coefficient.py:177](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/properties/wall_loss_coefficient.py#L177)

Calculate the wall loss coefficient for a rectangular chamber based on
the system state.

This function computes the wall loss coefficient for a rectangular chamber
using the system's physical state, including the wall eddy diffusivity,
particle properties (radius, density), and environmental conditions
(temperature, pressure). The chamber dimensions (length, width, height)
are also considered.

#### Arguments

- `wall_eddy_diffusivity` - The rate of wall eddy diffusivity in inverse
    seconds (s⁻¹).
- `particle_radius` - The radius of the particle in meters (m).
- `particle_density` - The density of the particle in kilograms per cubic
    meter (kg/m³).
- `temperature` - The temperature of the system in Kelvin (K).
- `pressure` - The pressure of the system in Pascals (Pa).
- `chamber_dimensions` - A tuple containing the length, width, and height
    of the rectangular chamber in meters (m).

#### Returns

The calculated wall loss coefficient for the rectangular chamber.

#### References

- Crump, J. G., & Seinfeld, J. H. (1981). TURBULENT DEPOSITION AND
    GRAVITATIONAL SEDIMENTATION OF AN AEROSOL IN A VESSEL OF ARBITRARY
    SHAPE. In J Aerosol Sct (Vol. 12, Issue 5).
    https://doi.org/10.1016/0021-8502(81)90036-7

#### Signature

```python
def rectangle_wall_loss_coefficient_via_system_state(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]: ...
```



## spherical_wall_loss_coefficient

[Show source in wall_loss_coefficient.py:28](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/properties/wall_loss_coefficient.py#L28)

Calculate the wall loss coefficient for a spherical chamber
approximation.

#### Arguments

- `wall_eddy_diffusivity` - Rate of the wall eddy diffusivity.
- `diffusion_coefficient` - Diffusion coefficient of the
    particle.
- `settling_velocity` - Settling velocity of the particle.
- `chamber_radius` - Radius of the chamber.

#### Returns

The calculated wall loss rate for a spherical chamber.

#### References

- Crump, J. G., Flagan, R. C., & Seinfeld, J. H. (1982). Particle wall
    loss rates in vessels. Aerosol Science and Technology, 2(3),
    303-309. https://doi.org/10.1080/02786828308958636

#### Signature

```python
def spherical_wall_loss_coefficient(
    wall_eddy_diffusivity: Union[float, NDArray[np.float64]],
    diffusion_coefficient: Union[float, NDArray[np.float64]],
    settling_velocity: Union[float, NDArray[np.float64]],
    chamber_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## spherical_wall_loss_coefficient_via_system_state

[Show source in wall_loss_coefficient.py:120](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/properties/wall_loss_coefficient.py#L120)

Calculate the wall loss coefficient for a spherical chamber based on the
system state.

This function computes the wall loss coefficient for a spherical chamber
using the system's physical state, including the wall eddy diffusivity,
particle properties (radius, density), and environmental conditions
(temperature, pressure). The chamber radius is also taken into account.

#### Arguments

- `wall_eddy_diffusivity` - The rate of wall eddy diffusivity in inverse
    seconds (s⁻¹).
- `particle_radius` - The radius of the particle in meters (m).
- `particle_density` - The density of the particle in kilograms per cubic
    meter (kg/m³).
- `temperature` - The temperature of the system in Kelvin (K).
- `pressure` - The pressure of the system in Pascals (Pa).
- `chamber_radius` - The radius of the spherical chamber in meters (m).

#### Returns

The calculated wall loss coefficient for the spherical chamber.

#### Signature

```python
def spherical_wall_loss_coefficient_via_system_state(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_radius: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
