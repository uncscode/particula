# Aerodynamic Mobility Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Aerodynamic Mobility Module

> Auto-generated documentation for [particula.next.particles.properties.aerodynamic_mobility_module](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/aerodynamic_mobility_module.py) module.

## particle_aerodynamic_mobility

[Show source in aerodynamic_mobility_module.py:9](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/aerodynamic_mobility_module.py#L9)

Calculate the aerodynamic mobility of a particle, defined as the ratio
of the slip correction factor to the product of the dynamic viscosity of
the fluid, the particle radius, and a slip correction constant derived.

This mobility quantifies the ease with which a particle can move through
a fluid.

#### Arguments

radius : The radius of the particle (m).
slip_correction_factor : The slip correction factor for the particle
    in the fluid (dimensionless).
dynamic_viscosity : The dynamic viscosity of the fluid (Pa.s).

#### Returns

The particle aerodynamic mobility (m^2/s).

#### Signature

```python
def particle_aerodynamic_mobility(
    radius: Union[float, NDArray[np.float64]],
    slip_correction_factor: Union[float, NDArray[np.float64]],
    dynamic_viscosity: float,
) -> Union[float, NDArray[np.float64]]: ...
```
