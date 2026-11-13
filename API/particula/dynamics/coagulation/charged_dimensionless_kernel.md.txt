# particula.dynamics.coagulation.charged_dimensionless_kernel

*module*

Dimensionless coagulation for the transition regime.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L1-L1)

## particula.dynamics.coagulation.charged_dimensionless_kernel.get_coulomb_kernel_chahl2019

*function*

```python
def get_coulomb_kernel_chahl2019(diffusive_knudsen: Union[float, NDArray[np.float64]], coulomb_potential_ratio: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Chahl and Gopalakrishnan (2019) dimensionless coagulation kernel.

This function accounts for the Coulomb potential between particles using
the Chahl and Gopalakrishnan (2019) approximation.

**Parameters**

- `diffusive_knudsen` (`Union[float, NDArray[np.float64]]`) — The diffusive Knudsen number (K_nD) [dimensionless].
- `coulomb_potential_ratio` (`Union[float, NDArray[np.float64]]`) — The Coulomb potential ratio (phi_E) [dimensionless].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The dimensionless coagulation kernel (H) [dimensionless].

> **References**
>
> - Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
>   molecular regime Coulombic collisions in aerosols and dusty plasmas.
>   Aerosol Science and Technology, 53(8), 933-957.
>   https://doi.org/10.1080/02786826.2019.1614522

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L279-L328)

## particula.dynamics.coagulation.charged_dimensionless_kernel.get_coulomb_kernel_dyachkov2007

*function*

```python
def get_coulomb_kernel_dyachkov2007(diffusive_knudsen: Union[float, NDArray[np.float64]], coulomb_potential_ratio: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Dyachkov et al. (2007) approximation for the dimensionless kernel.

This function accounts for the Coulomb potential between particles using
the Dyachkov et al. (2007) approximation.

**Parameters**

- `diffusive_knudsen` (`Union[float, NDArray[np.float64]]`) — The diffusive Knudsen number (K_nD) [dimensionless].
- `coulomb_potential_ratio` (`Union[float, NDArray[np.float64]]`) — The Coulomb potential ratio (phi_E) [dimensionless].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The dimensionless coagulation kernel (H) [dimensionless].

References:
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
  particles in the transition regime: The effect of the Coulomb potential.
  Journal of Chemical Physics, 126(12).
  https://doi.org/10.1063/1.2713719

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L109-L162)

## particula.dynamics.coagulation.charged_dimensionless_kernel.get_coulomb_kernel_gatti2008

*function*

```python
def get_coulomb_kernel_gatti2008(diffusive_knudsen: Union[float, NDArray[np.float64]], coulomb_potential_ratio: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Gatti et al. (2008) approximation for the dimensionless kernel.

This function accounts for the Coulomb potential between particles using
the Gatti et al. (2008) approximation.

**Parameters**

- `diffusive_knudsen` (`Union[float, NDArray[np.float64]]`) — The diffusive Knudsen number (K_nD) [dimensionless].
- `coulomb_potential_ratio` (`Union[float, NDArray[np.float64]]`) — The Coulomb potential ratio (phi_E) [dimensionless].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The dimensionless coagulation kernel (H) [dimensionless].

References:
- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
  charging in plasmas over a wide range of collisionality. Physical Review
  E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
  https://doi.org/10.1103/PhysRevE.78.046402

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L165-L230)

## particula.dynamics.coagulation.charged_dimensionless_kernel.get_coulomb_kernel_gopalakrishnan2012

*function*

```python
def get_coulomb_kernel_gopalakrishnan2012(diffusive_knudsen: Union[float, NDArray[np.float64]], coulomb_potential_ratio: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Gopalakrishnan and Hogan (2012) dimensionless coagulation kernel.

This function accounts for the Coulomb potential between particles using
the Gopalakrishnan and Hogan (2012) approximation.

**Parameters**

- `diffusive_knudsen` (`Union[float, NDArray[np.float64]]`) — The diffusive Knudsen number (K_nD) [dimensionless].
- `coulomb_potential_ratio` (`Union[float, NDArray[np.float64]]`) — The Coulomb potential ratio (phi_E) [dimensionless].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The dimensionless coagulation kernel (H) [dimensionless].

References:
- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
  in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear
  and Soft Matter Physics, 85(2).
  https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L233-L276)

## particula.dynamics.coagulation.charged_dimensionless_kernel.get_dimensional_kernel

*function*

```python
def get_dimensional_kernel(dimensionless_kernel: NDArray[np.float64], coulomb_potential_ratio: NDArray[np.float64], sum_of_radii: NDArray[np.float64], reduced_mass: NDArray[np.float64], reduced_friction_factor: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate the dimensioned coagulation kernel for each particle pair.

This function computes the dimensioned coagulation kernel from the
dimensionless coagulation kernel and the reduced quantities. All inputs
are square matrices, representing all particle-particle interactions.

**Parameters**

- `dimensionless_kernel` (`NDArray[np.float64]`) — The dimensionless coagulation kernel (H) [dimensionless].
- `coulomb_potential_ratio` (`NDArray[np.float64]`) — The Coulomb potential ratio [dimensionless].
- `sum_of_radii` (`NDArray[np.float64]`) — The sum of the radii of the particles [m].
- `reduced_mass` (`NDArray[np.float64]`) — The reduced mass of the particles [kg].
- `reduced_friction_factor` (`NDArray[np.float64]`) — The reduced friction factor of the particles [dimensionless].

**Returns**

- (`NDArray[np.float64]`) — The dimensioned coagulation kernel, as a square matrix, of all particle-particle interactions [m³/s].

> **References**
>
> - Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
>   molecular regime Coulombic collisions in aerosols and dusty plasmas.
>   Aerosol Science and Technology, 53(8), 933-957.
>   https://doi.org/10.1080/02786826.2019.1614522

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L12-L56)

## particula.dynamics.coagulation.charged_dimensionless_kernel.get_hard_sphere_kernel

*function*

```python
def get_hard_sphere_kernel(diffusive_knudsen: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Hard sphere approximation for the dimensionless coagulation kernel.

This function provides a hard sphere approximation for the dimensionless
coagulation kernel based on the diffusive Knudsen number.

**Parameters**

- `diffusive_knudsen` (`Union[float, NDArray[np.float64]]`) — The diffusive Knudsen number (K_nD) [dimensionless].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The dimensionless coagulation kernel (H) [dimensionless].

**Raises**

- `ValueError` — If diffusive_knudsen contains negative values, NaN, or
infinity.

References:
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
  particles in the transition regime: The effect of the Coulomb potential.
  Journal of Chemical Physics, 126(12).
  https://doi.org/10.1063/1.2713719

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensionless_kernel.py#L59-L106)
