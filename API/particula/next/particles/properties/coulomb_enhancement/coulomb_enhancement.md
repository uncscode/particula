# Coulomb Enhancement

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Coulomb Enhancement

> Auto-generated documentation for [particula.next.particles.properties.coulomb_enhancement](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/coulomb_enhancement.py) module.

## continuum

[Show source in coulomb_enhancement.py:89](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/coulomb_enhancement.py#L89)

Calculates the Coulombic enhancement continuum limit. For all particle-
particle interactions.

#### Arguments

- `coulomb_potential` - The Coulomb potential ratio [dimensionless].

#### Returns

The Coulomb enhancement for the continuum limit [dimensionless].

#### References

Equation 6b in: Gopalakrishnan, R., & Hogan, C. J. (2012).
Coulomb-influenced collisions in aerosols and dusty plasmas.
Physical Review E - Statistical, Nonlinear,
and Soft Matter Physics, 85(2).
(https://doi.org/10.1103/PhysRevE.85.026410)

#### Signature

```python
def continuum(
    coulomb_potential: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## kinetic

[Show source in coulomb_enhancement.py:61](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/coulomb_enhancement.py#L61)

Calculates the Coulombic enhancement kinetic limit. For all particle-
particle interactions.

#### Arguments

- `coulomb_potential` - The Coulomb potential ratio [dimensionless].

#### Returns

The Coulomb enhancement for the kinetic limit [dimensionless].

#### References

Equation 6d and 6e in, Gopalakrishnan, R., & Hogan, C. J. (2012).
Coulomb-influenced collisions in aerosols and dusty plasmas.
Physical Review E - Statistical, Nonlinear,
and Soft Matter Physics, 85(2).
(https://doi.org/10.1103/PhysRevE.85.026410)

#### Signature

```python
def kinetic(
    coulomb_potential: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## ratio

[Show source in coulomb_enhancement.py:22](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/coulomb_enhancement.py#L22)

Calculates the Coulomb potential ratio, phi_E. For all particle-
particle interactions.

#### Arguments

- `radius` - The radius of the particle [m].
- `charge` - The number of charges on the particle [dimensionless].
- `temperature` - The temperature of the system [K].

#### Returns

The Coulomb potential ratio [dimensionless].

#### References

- `Equation` *7* - Gopalakrishnan, R., & Hogan, C. J. (2012).
    Coulomb-influenced collisions in aerosols and dusty plasmas.
    Physical Review E - Statistical, Nonlinear, and Soft Matter
    Physics, 85(2). (https://doi.org/10.1103/PhysRevE.85.026410)

#### Signature

```python
def ratio(
    radius: Union[float, NDArray[np.float64]],
    charge: Union[int, NDArray[np.float64]] = 0,
    temperature: float = 298.15,
) -> Union[float, NDArray[np.float64]]: ...
```
