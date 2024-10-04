# Kernel

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Kernel

> Auto-generated documentation for [particula.next.dynamics.coagulation.kernel](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py) module.

## CoulombDyachkov2007

[Show source in kernel.py:130](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L130)

Dyachkov et al. (2007) approximation for the dimensionless coagulation
kernel. Accounts for the Coulomb potential between particles.

#### References

-----------
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
particles in the transition regime: The effect of the Coulomb potential.
Journal of Chemical Physics, 126(12).
https://doi.org/10.1063/1.2713719

#### Signature

```python
class CoulombDyachkov2007(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### CoulombDyachkov2007().dimensionless

[Show source in kernel.py:143](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L143)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## CoulombGatti2008

[Show source in kernel.py:153](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L153)

Gatti and Kortshagen (2008) approximation for the dimensionless coagulation
kernel. Accounts for the Coulomb potential between particles.

#### References

-----------
- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
charging in plasmas over a wide range of collisionality. Physical Review
E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
https://doi.org/10.1103/PhysRevE.78.046402

#### Signature

```python
class CoulombGatti2008(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### CoulombGatti2008().dimensionless

[Show source in kernel.py:166](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L166)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## CoulombGopalakrishnan2012

[Show source in kernel.py:176](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L176)

Gopalakrishnan and Hogan (2012) approximation for the dimensionless
coagulation kernel. Accounts for the Coulomb potential between particles.

#### References

-----------
- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear,
and Soft Matter Physics, 85(2).
https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
class CoulombGopalakrishnan2012(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### CoulombGopalakrishnan2012().dimensionless

[Show source in kernel.py:189](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L189)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## CoulumbChahl2019

[Show source in kernel.py:199](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L199)

Chahl and Gopalakrishnan (2019) approximation for the dimensionless
coagulation kernel. Accounts for the Coulomb potential between particles.

#### References

-----------
- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
molecular regime Coulombic collisions in aerosols and dusty plasmas.
Aerosol Science and Technology, 53(8), 933-957.
https://doi.org/10.1080/02786826.2019.1614522

#### Signature

```python
class CoulumbChahl2019(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### CoulumbChahl2019().dimensionless

[Show source in kernel.py:212](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L212)

#### Signature

```python
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```



## HardSphere

[Show source in kernel.py:117](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L117)

Hard sphere dimensionless coagulation strategy.

#### Signature

```python
class HardSphere(KernelStrategy): ...
```

#### See also

- [KernelStrategy](#kernelstrategy)

### HardSphere().dimensionless

[Show source in kernel.py:122](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L122)

#### Signature

```python
def dimensionless(
    self, diffusive_knudsen: NDArray[np.float64], coulomb_potential_ratio: ignore
) -> NDArray[np.float64]: ...
```



## KernelStrategy

[Show source in kernel.py:12](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L12)

Abstract class for dimensionless coagulation strategies. This class defines
the dimensionless kernel (H) method that must be implemented by any
dimensionless coagulation strategy.

#### Methods

--------
- dimensionless (abstractmethod): Calculate the dimensionless coagulation
kernel.
- `-` *kernel* - Calculate the dimensioned coagulation kernel.

#### Signature

```python
class KernelStrategy(ABC): ...
```

### KernelStrategy().dimensionless

[Show source in kernel.py:25](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L25)

Return the dimensionless coagulation kernel (H)

#### Arguments

-----
- `-` *diffusive_knudsen* - The diffusive Knudsen number (K_nD)
[dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio (phi_E)
[dimensionless].

#### Returns

--------
The dimensionless coagulation kernel (H) [dimensionless].

#### References

-----------
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation
of particles in the transition regime: The effect of the Coulomb
potential. Journal of Chemical Physics, 126(12).
https://doi.org/10.1063/1.2713719
- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
charging in plasmas over a wide range of collisionality. Physical
Review E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
https://doi.org/10.1103/PhysRevE.78.046402
- Gopalakrishnan, R., & Hogan, C. J. (2011). Determination of the
transition regime collision kernel from mean first passage times.
Aerosol Science and Technology, 45(12), 1499-1509.
https://doi.org/10.1080/02786826.2011.601775
- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced
collisions in aerosols and dusty plasmas. Physical Review E -
Statistical, Nonlinear, and Soft Matter Physics, 85(2).
https://doi.org/10.1103/PhysRevE.85.026410
- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
molecular regime Coulombic collisions in aerosols and dusty plasmas.
Aerosol Science and Technology, 53(8), 933-957.
https://doi.org/10.1080/02786826.2019.1614522

#### Signature

```python
@abstractmethod
def dimensionless(
    self,
    diffusive_knudsen: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```

### KernelStrategy().kernel

[Show source in kernel.py:68](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/kernel.py#L68)

The dimensioned coagulation kernel for each particle pair, calculated
from the dimensionless coagulation kernel and the reduced quantities.
All inputs are square matrices, for all particle-particle interactions.

#### Arguments

-----
- `-` *dimensionless_kernel* - The dimensionless coagulation kernel
[dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio [dimensionless].
- `-` *sum_of_radii* - The sum of the radii of the particles [m].
- `-` *reduced_mass* - The reduced mass of the particles [kg].
- `-` *reduced_friction_factor* - The reduced friction factor of the
particles [dimensionless].

#### Returns

--------
The dimensioned coagulation kernel, as a square matrix, of all
particle-particle interactions [m^3/s].

Check, were the /s comes from.

#### References

-----------

#### Signature

```python
def kernel(
    self,
    dimensionless_kernel: NDArray[np.float64],
    coulomb_potential_ratio: NDArray[np.float64],
    sum_of_radii: NDArray[np.float64],
    reduced_mass: NDArray[np.float64],
    reduced_friction_factor: NDArray[np.float64],
) -> NDArray[np.float64]: ...
```
