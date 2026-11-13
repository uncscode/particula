# particula.dynamics.coagulation.brownian_kernel

*module*

The basic Brownian coagulation kernel for aerosol particles.

This module provides functions to calculate the Brownian coagulation
kernel for aerosol particles, based on Fuchs' theory as described by
Seinfeld and Pandis (2016), Chapter 13, Table 13.1.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L1-L1)

## particula.dynamics.coagulation.brownian_kernel._brownian_diffusivity

*function*

```python
def _brownian_diffusivity(temperature: Union[float, NDArray[np.float64]], aerodynamic_mobility: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the diffusivity of particles due to Brownian motion.

This function calculates the diffusivity of particles due to Brownian
motion, which is essentially the scaled aerodynamic mobility of the
particles. The equation used is:

- D = k × T × B
    - D is the diffusivity of the particles [m²/s].
    - k is the Boltzmann constant [J/K].
    - T is the temperature of the air [K].
    - B is the aerodynamic mobility of the particles [m²/s].

**Parameters**

- `temperature` (`Union[float, NDArray[np.float64]]`) — The temperature of the air [K].
- `aerodynamic_mobility` (`Union[float, NDArray[np.float64]]`) — The aerodynamic mobility of the particles [m²/s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The diffusivity of the particles due to Brownian motion [m²/s].

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
>   physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
>   Coefficient K12.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L253-L282)

## particula.dynamics.coagulation.brownian_kernel._g_collection_term

*function*

```python
def _g_collection_term(mean_free_path_particle: Union[float, NDArray[np.float64]], particle_radius: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the `g` collection term for Brownian coagulation.

This function calculates the `g` collection term for Brownian
coagulation, defined as the ratio of the mean free path of the particles
to the radius of the particles. The equation used is:

- g = ((2r + λ)³ - (4r² + λ²)^(3/2)) / (6rλ) - 2r
    - g is the collection term for Brownian coagulation [dimensionless].
    - λ is the mean free path of the particles [m].
    - r is the radius of the particles [m].

**Parameters**

- `mean_free_path_particle` (`Union[float, NDArray[np.float64]]`) — The mean free path of the particles [m].
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The collection term for Brownian coagulation [dimensionless].

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
>   physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
>   Coefficient K12.

> **Note**
>
> The np.sqrt(2) term appears to be an error in the text, as the term is
> not used in the second edition of the book. When it is used, the values
> are too small, by about 2x.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L215-L250)

## particula.dynamics.coagulation.brownian_kernel._mean_free_path_l

*function*

```python
def _mean_free_path_l(diffusivity_particle: Union[float, NDArray[np.float64]], mean_thermal_speed_particle: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the mean free path of particles for coagulation.

Calculate the mean free path of particles for coagulation.

This function calculates the mean free path of particles, defined for
Brownian coagulation as the ratio of the diffusivity of the particles
to their mean thermal speed. This parameter is crucial for understanding
particle dynamics in a fluid. The equation used is:

- λ = (8 × D) / (π × v)
    - λ is the mean free path of the particles [m].
    - D is the diffusivity of the particles [m²/s].
    - v is the mean thermal speed of the particles [m/s].

**Parameters**

- `diffusivity_particle` (`Union[float, NDArray[np.float64]]`) — The diffusivity of the particles [m²/s].
- `mean_thermal_speed_particle` (`Union[float, NDArray[np.float64]]`) — The mean thermal speed of the particles [m/s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The mean free path of the particles [m].

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
>   physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
>   Coefficient K12.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L181-L212)

## particula.dynamics.coagulation.brownian_kernel.get_brownian_kernel

*function*

```python
def get_brownian_kernel(particle_radius: Union[float, NDArray[np.float64]], diffusivity_particle: Union[float, NDArray[np.float64]], g_collection_term_particle: Union[float, NDArray[np.float64]], mean_thermal_speed_particle: Union[float, NDArray[np.float64]], alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0) -> Union[float, NDArray[np.float64]]
```

Calculate the Brownian coagulation kernel for aerosol particles.

This function computes the Brownian coagulation kernel, which is
defined as the product of the diffusivity of the particles, the
collection term `g`, and the radius of the particles. The equation
used is:

- K = (4π × D × r) / (r / (r + g) + 4D / (r × v × α))
    - K is the Brownian coagulation kernel [m³/s].
    - D is the diffusivity of the particles [m²/s].
    - r is the radius of the particles [m].
    - g is the collection term for Brownian coagulation [dimensionless].
    - v is the mean thermal speed of the particles [m/s].
    - α is the collision efficiency of the particles [dimensionless].

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].
- `diffusivity_particle` (`Union[float, NDArray[np.float64]]`) — The diffusivity of the particles [m²/s].
- `g_collection_term_particle` (`Union[float, NDArray[np.float64]]`) — The collection term for Brownian coagulation [dimensionless].
- `mean_thermal_speed_particle` (`Union[float, NDArray[np.float64]]`) — The mean thermal speed of the particles [m/s].
- `alpha_collision_efficiency` (`Union[float, NDArray[np.float64]]`) (default: `1.0`) — The collision efficiency of the particles [dimensionless].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Square matrix of Brownian coagulation kernel for aerosol particles [m³/s].

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
>   physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
>   Coefficient K12 (with alpha collision efficiency term 13.56).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L17-L96)

## particula.dynamics.coagulation.brownian_kernel.get_brownian_kernel_via_system_state

*function*

```python
def get_brownian_kernel_via_system_state(particle_radius: Union[float, NDArray[np.float64]], particle_mass: Union[float, NDArray[np.float64]], temperature: float, pressure: float, alpha_collision_efficiency: Union[float, NDArray[np.float64]] = 1.0) -> Union[float, NDArray[np.float64]]
```

Calculate the Brownian coagulation kernel using system state parameters.

This function calculates the Brownian coagulation kernel for aerosol
particles by determining the necessary intermediate properties such as
particle diffusivity and mean thermal speed. The equation used is:

- K = (4π × D × r) / (r / (r + g) + 4D / (r × v × α))
    - K is the Brownian coagulation kernel [m³/s].
    - D is the diffusivity of the particles [m²/s].
    - r is the radius of the particles [m].
    - g is the collection term for Brownian coagulation [dimensionless].
    - v is the mean thermal speed of the particles [m/s].
    - α is the collision efficiency of the particles [dimensionless].

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particles [m].
- `particle_mass` (`Union[float, NDArray[np.float64]]`) — The mass of the particles [kg].
- `temperature` (`float`) — The temperature of the air [K].
- `pressure` (`float`) — The pressure of the air [Pa].
- `alpha_collision_efficiency` (`Union[float, NDArray[np.float64]]`) (default: `1.0`) — The collision efficiency of the particles [dimensionless].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Square matrix of Brownian coagulation kernel for aerosol particles [m³/s].

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric chemistry and
>   physics, Section 13 TABLE 13.1 Fuchs Form of the Brownian Coagulation
>   Coefficient K12.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/brownian_kernel.py#L99-L178)
