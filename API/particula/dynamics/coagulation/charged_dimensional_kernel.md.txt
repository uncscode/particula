# particula.dynamics.coagulation.charged_dimensional_kernel

*module*

Charged dimensional kernel for coagulation calculated from system state.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L1-L1)

## particula.dynamics.coagulation.charged_dimensional_kernel._system_state_properties

*function*

```python
def _system_state_properties(particle_radius: Union[float, NDArray[np.float64]], particle_mass: Union[float, NDArray[np.float64]], particle_charge: Union[float, NDArray[np.float64]], temperature: float, pressure: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
```

Get the system state properties for charged particles.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].
- `particle_mass` (`Union[float, NDArray[np.float64]]`) — The mass of the particles [kg].
- `particle_charge` (`Union[float, NDArray[np.float64]]`) — The charge of the particles [C].
- `temperature` (`float`) — The temperature of the system [K].
- `pressure` (`float`) — The pressure of the system [Pa].

**Returns**

- `coulomb_potential_ratio` (`NDArray[np.float64]`) — The Coulomb potential ratio [dimensionless].
- `diffusive_knudsen` (`NDArray[np.float64]`) — The diffusive knudsen number [dimensionless].
- `sum_of_radii` (`NDArray[np.float64]`) — The sum of the radii of the particles [m].
- `reduced_mass` (`NDArray[np.float64]`) — The reduced mass of the particles [kg].
- `reduced_friction_factor` (`NDArray[np.float64]`) — The reduced friction factor of the particles [dimensionless].

Examples: (This is an internal helper and typically not called directly.)

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L32-L121)

## particula.dynamics.coagulation.charged_dimensional_kernel.get_coulomb_kernel_chahl2019_via_system_state

*function*

```python
def get_coulomb_kernel_chahl2019_via_system_state(particle_radius: Union[float, NDArray[np.float64]], particle_mass: Union[float, NDArray[np.float64]], particle_charge: Union[float, NDArray[np.float64]], temperature: float, pressure: float) -> NDArray[np.float64]
```

Chahl (2019) dimensioned coagulation kernel via system state.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].
- `particle_mass` (`Union[float, NDArray[np.float64]]`) — The mass of the particles [kg].
- `particle_charge` (`Union[float, NDArray[np.float64]]`) — The charge of the particles [C].
- `temperature` (`float`) — The temperature of the system [K].
- `pressure` (`float`) — The pressure of the system [Pa].

**Returns**

- (`NDArray[np.float64]`) — The dimensioned coagulation kernel, as a square matrix, [m^3/s].

**Examples**

``` py title="Example Usage"
import particula as par
kernel_chahl = ()
    par.dynamics.get_coulomb_kernel_chahl2019_via_system_state(
    p_radius, p_mass, p_charge, 298.15, 101325
    )
)
print(kernel_chahl)
```

References:
- Chahl, H. S., & Gopalakrishnan, R. (2019). High potential, near free
    molecular regime Coulombic collisions in aerosols and dusty plasmas.
    Aerosol Science and Technology, 53(8), 933-957.
    https://doi.org/10.1080/02786826.2019.1614522

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L402-L466)

## particula.dynamics.coagulation.charged_dimensional_kernel.get_coulomb_kernel_dyachkov2007_via_system_state

*function*

```python
def get_coulomb_kernel_dyachkov2007_via_system_state(particle_radius: Union[float, NDArray[np.float64]], particle_mass: Union[float, NDArray[np.float64]], particle_charge: Union[float, NDArray[np.float64]], temperature: float, pressure: float) -> NDArray[np.float64]
```

The dimensioned coagulation kernel via Dyachkov (2007).

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].
- `particle_mass` (`Union[float, NDArray[np.float64]]`) — The mass of the particles [kg].
- `particle_charge` (`Union[float, NDArray[np.float64]]`) — The charge of the particles [C].
- `temperature` (`float`) — The temperature of the system [K].
- `pressure` (`float`) — The pressure of the system [Pa].

**Returns**

- (`NDArray[np.float64]`) — The dimensioned coagulation kernel, as a square matrix, [m^3/s].

**Examples**

``` py title="Example Usage"
import particula as par
kernel_dyachkov = (
    par.dynamics.get_coulomb_kernel_dyachkov2007_via_system_state(
        p_radius, p_mass, p_charge, 298.15, 101325
    )
)
print(kernel_dyachkov.shape)
```

References:
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
  particles in the transition regime: The effect of the Coulomb potential.
  Journal of Chemical Physics, 126(12).
  https://doi.org/10.1063/1.2713719

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L201-L265)

## particula.dynamics.coagulation.charged_dimensional_kernel.get_coulomb_kernel_gatti2008_via_system_state

*function*

```python
def get_coulomb_kernel_gatti2008_via_system_state(particle_radius: Union[float, NDArray[np.float64]], particle_mass: Union[float, NDArray[np.float64]], particle_charge: Union[float, NDArray[np.float64]], temperature: float, pressure: float) -> NDArray[np.float64]
```

The dimensioned coagulation kernel via system state using Gatti (2008).

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].
- `particle_mass` (`Union[float, NDArray[np.float64]]`) — The mass of the particles [kg].
- `particle_charge` (`Union[float, NDArray[np.float64]]`) — The charge of the particles [C].
- `temperature` (`float`) — The temperature of the system [K].
- `pressure` (`float`) — The pressure of the system [Pa].

**Returns**

- (`NDArray[np.float64]`) — The dimensioned coagulation kernel, as a square matrix, [m^3/s].

**Examples**

``` py title="Example Usage"
import particula as par
kernel_gatti = (
    par.dynamics.get_coulomb_kernel_gatti2008_via_system_state(
        p_radius, p_mass, p_charge, 298.15, 101325
    )
)
print(kernel_gatti)
```

References:
- Gatti, M., & Kortshagen, U. (2008). Analytical model of particle
  charging in plasmas over a wide range of collisionality. Physical Review
  E - Statistical, Nonlinear, and Soft Matter Physics, 78(4).
  https://doi.org/10.1103/PhysRevE.78.046402

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L268-L332)

## particula.dynamics.coagulation.charged_dimensional_kernel.get_coulomb_kernel_gopalakrishnan2012_via_system_state

*function*

```python
def get_coulomb_kernel_gopalakrishnan2012_via_system_state(particle_radius: Union[float, NDArray[np.float64]], particle_mass: Union[float, NDArray[np.float64]], particle_charge: Union[float, NDArray[np.float64]], temperature: float, pressure: float) -> NDArray[np.float64]
```

Gopalakrishnan (2012) dimensioned coagulation kernel via system state.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].
- `particle_mass` (`Union[float, NDArray[np.float64]]`) — The mass of the particles [kg].
- `particle_charge` (`Union[float, NDArray[np.float64]]`) — The charge of the particles [C].
- `temperature` (`float`) — The temperature of the system [K].
- `pressure` (`float`) — The pressure of the system [Pa].

**Returns**

- (`NDArray[np.float64]`) — The dimensioned coagulation kernel, as a square matrix, [m^3/s].

**Examples**

``` py title="Example Usage"
import particula as par
kernel_gopal = (
    par.dyanmics.get_coulomb_kernel_gopalakrishnan2012_via_system_state(
        p_radius, p_mass, p_charge, 298.15, 101325
    )
)
# kernel_gopal is an N×N matrix [m^3/s]
```

References:
- Gopalakrishnan, R., & Hogan, C. J. (2012). Coulomb-influenced collisions
  in aerosols and dusty plasmas. Physical Review E - Statistical, Nonlinear
  and Soft Matter Physics, 85(2).
  https://doi.org/10.1103/PhysRevE.85.026410

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L335-L399)

## particula.dynamics.coagulation.charged_dimensional_kernel.get_hard_sphere_kernel_via_system_state

*function*

```python
def get_hard_sphere_kernel_via_system_state(particle_radius: Union[float, NDArray[np.float64]], particle_mass: Union[float, NDArray[np.float64]], particle_charge: Union[float, NDArray[np.float64]], temperature: float, pressure: float) -> NDArray[np.float64]
```

The hard sphere dimensioned coagulation kernel via system state.

For the hard sphere kernel, the dimensionless kernel is computed
internally based on the diffusive Knudsen number, then converted
to the dimensioned form using the system state properties
(particle radius, mass, charge, temperature, pressure).
coulomb potential ratio, sum of radii, reduced mass...etc are all
calculated from the system state properties (temperature, pressure, etc.).
These are used to calculate the dimensionless kernel, which is then
converted to the dimensioned kernel.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].
- `particle_mass` (`Union[float, NDArray[np.float64]]`) — The mass of the particles [kg].
- `particle_charge` (`Union[float, NDArray[np.float64]]`) — The charge of the particles [C].
- `temperature` (`float`) — The temperature of the system [K].
- `pressure` (`float`) — The pressure of the system [Pa].

**Returns**

- (`NDArray[np.float64]`) — The dimensioned coagulation kernel, as a square matrix, of all particle-particle interactions [m^3/s].

**Examples**

``` py title="Example Usage"
kernel_matrix = get_hard_sphere_kernel_via_system_state(
    p_radius, p_mass, p_charge, 298.15, 101325
)
# kernel_matrix is an N×N array of rates
```

References:
- Dyachkov, S. A., Kustova, E. V., & Kustov, A. V. (2007). Coagulation of
  particles in the transition regime: The effect of the Coulomb potential.
  Journal of Chemical Physics, 126(12).
  https://doi.org/10.1063/1.2713719

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/charged_dimensional_kernel.py#L124-L198)
