# Aerodynamic Size

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Aerodynamic Size

> Auto-generated documentation for [particula.next.particles.properties.aerodynamic_size](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/aerodynamic_size.py) module.

## get_aerodynamic_shape_factor

[Show source in aerodynamic_size.py:68](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/aerodynamic_size.py#L68)

Retrieve the aerodynamic shape factor for a given particle shape.

#### Arguments

- `shape_key` - The shape of the particle as a string.

#### Returns

The shape factor of the particle as a float.

#### Raises

- `ValueError` - If the shape is not found in the predefined shape
    factor dictionary.

#### Signature

```python
def get_aerodynamic_shape_factor(shape_key: str) -> float: ...
```



## particle_aerodynamic_length

[Show source in aerodynamic_size.py:24](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/aerodynamic_size.py#L24)

Calculate the aerodynamic length scale of a particle.

The aerodynamic size is used to compare the aerodynamic properties of
particles with their physical properties, particularly when interpreting
aerodynamic particle sizer measurements. This is either the aerodynamic
diameter or radius, depending on length scale input.

#### Arguments

- `physical_length` - Physical length scale of the particle (m).
- `physical_slip_correction_factor` - Slip correction factor for the
    particle's physical radius in the fluid (dimensionless).
- `aerodynamic_slip_correction_factor` - Slip correction factor for the
    particle's aerodynamic radius in the fluid (dimensionless).
- `density` - Density of the particle (kg/m^3).
- `reference_density` - Reference density for the particle, typically the
    density of water (1000 kg/m^3 by default).
- `aerodynamic_shape_factor` - Shape factor of the particle, accounting for
    non-sphericity (dimensionless, default is 1.0 for spherical
    particles).

#### Returns

Aerodynamic radius of the particle (m).

#### References

- https://en.wikipedia.org/wiki/Aerosol#Aerodynamic_diameter
- Hinds, W.C. (1998) Aerosol Technology: Properties, behavior, and
    measurement of airborne particles. Wiley-Interscience, New York.
    pp 51-53, section 3.6.

#### Signature

```python
def particle_aerodynamic_length(
    physical_length: Union[float, NDArray[np.float64]],
    physical_slip_correction_factor: Union[float, NDArray[np.float64]],
    aerodynamic_slip_correction_factor: Union[float, NDArray[np.float64]],
    density: Union[float, NDArray[np.float64]],
    reference_density: float = 1000,
    aerodynamic_shape_factor: float = 1.0,
) -> Union[float, NDArray[np.float64]]: ...
```
