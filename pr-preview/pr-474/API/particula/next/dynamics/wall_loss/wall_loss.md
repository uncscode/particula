# Wall Loss

[Particula Index](../../../README.md#particula-index) / [Particula](../../index.md#particula) / [Next](../index.md#next) / [Dynamics](./index.md#dynamics) / Wall Loss

> Auto-generated documentation for [particula.next.dynamics.wall_loss](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/wall_loss.py) module.

## rectangle_wall_loss_rate

[Show source in wall_loss.py:65](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/wall_loss.py#L65)

Calculate the wall loss rate of particles in a rectangular chamber.

This function computes the rate at which particles are lost to the walls
of a rectangular chamber, based on the system state. It uses the wall eddy
diffusivity, particle properties (radius, density, concentration), and
environmental conditions (temperature, pressure) to determine the loss
rate. The chamber dimensions (length, width, height) are also taken
into account.

#### Arguments

- `wall_eddy_diffusivity` - The rate of wall eddy diffusivity in inverse
    seconds (s⁻¹).
- `particle_radius` - The radius of the particle in meters (m).
- `particle_density` - The density of the particle in kilograms per cubic
    meter (kg/m³).
- `particle_concentration` - The concentration of particles in the chamber
    in particles per cubic meter (#/m³).
- `temperature` - The temperature of the system in Kelvin (K).
- `pressure` - The pressure of the system in Pascals (Pa).
- `chamber_dimensions` - A tuple containing the length, width, and height
    of the rectangular chamber in meters (m).

#### Returns

The wall loss rate of the particles in the chamber.

#### Signature

```python
def rectangle_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_dimensions: Tuple[float, float, float],
) -> Union[float, NDArray[np.float64]]: ...
```



## spherical_wall_loss_rate

[Show source in wall_loss.py:16](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/wall_loss.py#L16)

Calculate the wall loss rate of particles in a spherical chamber.

This function computes the rate at which particles are lost to the walls
of a spherical chamber, based on the system state. It uses the wall eddy
diffusivity, particle properties (radius, density, concentration), and
environmental conditions (temperature, pressure) to determine the loss
rate.

#### Arguments

- `wall_eddy_diffusivity` - The rate of wall eddy diffusivity in inverse
    seconds (s⁻¹).
- `particle_radius` - The radius of the particle in meters (m).
- `particle_density` - The density of the particle in kilograms per cubic
    meter (kg/m³).
- `particle_concentration` - The concentration of particles in the chamber
    in particles per cubic meter (#/m³).
- `temperature` - The temperature of the system in Kelvin (K).
- `pressure` - The pressure of the system in Pascals (Pa).
- `chamber_radius` - The radius of the spherical chamber in meters (m).

#### Returns

The wall loss rate of the particles in the chamber.

#### Signature

```python
def spherical_wall_loss_rate(
    wall_eddy_diffusivity: float,
    particle_radius: Union[float, NDArray[np.float64]],
    particle_density: Union[float, NDArray[np.float64]],
    particle_concentration: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
    chamber_radius: float,
) -> Union[float, NDArray[np.float64]]: ...
```
