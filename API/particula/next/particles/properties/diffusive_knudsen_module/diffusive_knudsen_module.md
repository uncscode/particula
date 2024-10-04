# Diffusive Knudsen Module

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Particles](../index.md#particles) / [Properties](./index.md#properties) / Diffusive Knudsen Module

> Auto-generated documentation for [particula.next.particles.properties.diffusive_knudsen_module](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/diffusive_knudsen_module.py) module.

## diffusive_knudsen_number

[Show source in diffusive_knudsen_module.py:13](https://github.com/uncscode/particula/blob/main/particula/next/particles/properties/diffusive_knudsen_module.py#L13)

Diffusive Knudsen number. The *diffusive* Knudsen number is different
from Knudsen number. Ratio of: mean persistence of one particle to the
effective length scale of particle--particle Coulombic interaction

#### Arguments

-----
- `-` *radius* - The radius of the particle [m].
- `-` *mass_particle* - The mass of the particle [kg].
- `-` *friction_factor* - The friction factor of the particle [dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio, zero if
 no charges [dimensionless].
- `-` *temperature* - The temperature of the system [K].

#### Returns

--------
The diffusive Knudsen number [dimensionless], as a square matrix, of all
particle-particle interactions.

#### References

-----------
- Equation 5 in, with charges:
Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
molecular regime Coulombic collisions in aerosols and dusty plasmas.
Aerosol Science and Technology, 53(8), 933-957.
https://doi.org/10.1080/02786826.2019.1614522
- Equation 3b in, no charges:
Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
in aerosols and dusty plasmas. Physical Review E - Statistical,
Nonlinear, and Soft Matter Physics, 85(2).
https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
def diffusive_knudsen_number(
    radius: Union[float, NDArray[np.float64]],
    mass_particle: Union[float, NDArray[np.float64]],
    friction_factor: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]] = 0.0,
    temperature: float = 298.15,
) -> Union[float, NDArray[np.float64]]: ...
```
