# Transition Regime

[Particula Index](../../../../README.md#particula-index) / [Particula](../../../index.md#particula) / [Next](../../index.md#next) / [Dynamics](../index.md#dynamics) / [Coagulation](./index.md#coagulation) / Transition Regime

> Auto-generated documentation for [particula.next.dynamics.coagulation.transition_regime](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/transition_regime.py) module.

## coulomb_chahl2019

[Show source in transition_regime.py:209](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/transition_regime.py#L209)

Chahl and Gopalakrishnan (2019) approximation for the dimensionless
coagulation kernel. Accounts for the Coulomb potential between particles.

#### Arguments

-----
- `-` *diffusive_knudsen* - The diffusive Knudsen number (K_nD) [dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio (phi_E)
[dimensionless].

#### Returns

--------
The dimensionless coagulation kernel (H) [dimensionless].

#### References

-----------
- Equations X in:
Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
molecular regime Coulombic collisions in aerosols and dusty plasmas.
Aerosol Science and Technology, 53(8), 933-957.
https://doi.org/10.1080/02786826.2019.1614522

#### Signature

```python
def coulomb_chahl2019(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## coulomb_dyachkov2007

[Show source in transition_regime.py:51](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/transition_regime.py#L51)

Dyachkov et al. (2007) approximation for the dimensionless coagulation
kernel. Accounts for the Coulomb potential between particles.

#### Arguments

-----
- `-` *diffusive_knudsen* - The diffusive Knudsen number (K_nD) [dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio (phi_E)
[dimensionless].

#### Returns

--------
The dimensionless coagulation kernel (H) [dimensionless].

#### References

-----------
Equations X in:
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
particles in the transition regime: The effect of the Coulomb potential.
Journal of Chemical Physics, 126(12).
https://doi.org/10.1063/1.2713719

#### Signature

```python
def coulomb_dyachkov2007(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## coulomb_gatti2008

[Show source in transition_regime.py:106](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/transition_regime.py#L106)

Gatti et al. (2008) approximation for the dimensionless coagulation
kernel. Accounts for the Coulomb potential between particles.

#### Arguments

-----
- `-` *diffusive_knudsen* - The diffusive Knudsen number (K_nD) [dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio (phi_E)
[dimensionless].

#### Returns

--------
The dimensionless coagulation kernel (H) [dimensionless].

#### References

-----------
- Equations X in:
Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
charging in plasmas over a wide range of collisionality. Physical Review
E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
https://doi.org/10.1103/PhysRevE.78.046402

#### Signature

```python
def coulomb_gatti2008(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## coulomb_gopalakrishnan2012

[Show source in transition_regime.py:170](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/transition_regime.py#L170)

Gopalakrishnan and Hogan (2012) approximation for the dimensionless
coagulation kernel. Accounts for the Coulomb potential between particles.

#### Arguments

-----
- `-` *diffusive_knudsen* - The diffusive Knudsen number (K_nD) [dimensionless].
- `-` *coulomb_potential_ratio* - The Coulomb potential ratio (phi_E)
[dimensionless].

#### Returns

--------
The dimensionless coagulation kernel (H) [dimensionless].

#### References

-----------
- Equations X in:
Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear,
and Soft Matter Physics, 85(2).
https://doi.org/10.1103/PhysRevE.85.026410

#### Signature

```python
def coulomb_gopalakrishnan2012(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
    coulomb_potential_ratio: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```



## hard_sphere

[Show source in transition_regime.py:12](https://github.com/uncscode/particula/blob/main/particula/next/dynamics/coagulation/transition_regime.py#L12)

Hard sphere approximation for the dimensionless coagulation kernel.

#### Arguments

-----
- `-` *diffusive_knudsen* - The diffusive Knudsen number (K_nD) [dimensionless].

#### Returns

--------
The dimensionless coagulation kernel (H) [dimensionless].

#### References

-----------
Equations X in:
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
particles in the transition regime: The effect of the Coulomb potential.
Journal of Chemical Physics, 126(12).
https://doi.org/10.1063/1.2713719

#### Signature

```python
def hard_sphere(
    diffusive_knudsen: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]: ...
```
