# Diffusion Coefficient

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Diffusion Coefficient

> Auto-generated documentation for [particula.next.particles.properties.diffusion_coefficient](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/diffusion_coefficient.py) module.

## particle_diffusion_coefficient

[Show source in diffusion_coefficient.py:25](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/diffusion_coefficient.py#L25)

Calculate the diffusion coefficient of a particle.

#### Arguments

- `temperature` - The temperature at which the particle is
    diffusing, in Kelvin. Defaults to 298.15 K.
- `boltzmann_constant` - The Boltzmann constant. Defaults to the
    standard value of 1.380649 x 10^-23 J/K.
- `aerodynamic_mobility` - The aerodynamic mobility of
    the particle [m^2/s].

#### Returns

The diffusion coefficient of the particle [m^2/s].

#### Signature

```python
def particle_diffusion_coefficient(
    temperature: Union[float, NDArray[np.float64]],
    aerodynamic_mobility: Union[float, NDArray[np.float64]],
    boltzmann_constant: float = BOLTZMANN_CONSTANT.m,
) -> Union[float, NDArray[np.float64]]: ...
```



## particle_diffusion_coefficient_via_system_state

[Show source in diffusion_coefficient.py:49](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/diffusion_coefficient.py#L49)

Calculate the diffusion coefficient of a particle.

#### Arguments

- `temperature` - The temperature of the system in Kelvin (K).
- `particle_radius` - The radius of the particle in meters (m).
- `pressure` - The pressure of the system in Pascals (Pa).

#### Returns

The diffusion coefficient of the particle in square meters per
second (m²/s).

#### Signature

```python
def particle_diffusion_coefficient_via_system_state(
    particle_radius: Union[float, NDArray[np.float64]],
    temperature: float,
    pressure: float,
) -> Union[float, NDArray[np.float64]]: ...
```
