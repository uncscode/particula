# particula.dynamics.condensation.mass_transfer

*module*

Particle Vapor Equilibrium, condensation, and evaporation based on partial
pressures to calculate dm/dt or other forms of particle growth and decay.

> **Equation**
>
> - dm/dt = 4π × radius × Di × Mi × f(Kn, α) × delta_pi / (R × T)
>   - radius : The particle radius in m.
>   - Di : The diffusion coefficient of species i in m²/s.
>   - Mi : The molar mass of species i in kg/mol.
>   - f(Kn, α) : Transition function based on Knudsen number and
>       accommodation coefficient.
>   - delta_pi : Difference in partial pressures between gas and particle
>       phases in Pa.
>   - R : Gas constant in J/(mol·K).
>   - T : Temperature in K.

> **References**
>
> - Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric Chemistry and
>   Physics: From Air Pollution to Climate Change (3rd ed.). John
>   Wiley & Sons, Inc.
> - Topping, D., & Bane, M. (2022). Introduction to Aerosol Modelling
>   (D. Topping & M. Bane, Eds.). Wiley. https://doi.org/10.1002/9781119625728
> - Aerosol Modeling: Chapter 2, Equation 2.40

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L1-L1)

## particula.dynamics.condensation.mass_transfer.get_first_order_mass_transport_k

*function*

```python
def get_first_order_mass_transport_k(particle_radius: Union[float, NDArray[np.float64]], vapor_transition: Union[float, NDArray[np.float64]], diffusion_coefficient: Union[float, NDArray[np.float64]] = 2e-05) -> Union[float, NDArray[np.float64]]
```

Calculate the first-order mass transport coefficient per particle.

This function computes the coefficient K that governs how fast mass is
transported to or from a particle in a vapor. The equation is:

- K = 4π × radius × D × X
    - K : Mass transport coefficient [m³/s].
    - radius : Particle radius [m].
    - D : Diffusion coefficient of the vapor [m²/s].
    - X : Vapor transition correction factor [unitless].

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `vapor_transition` (`Union[float, NDArray[np.float64]]`) — The vapor transition correction factor [unitless].
- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) (default: `2e-05`) — The diffusion coefficient of the vapor [m²/s]. Defaults to 2e-5 (approx. air).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The first-order mass transport coefficient per particle [m³/s].

**Examples**

```py title="Float input"
import particula as par
par.dynamics.get_first_order_mass_transport_k(
    particle_radius=1e-6,
    vapor_transition=0.6,
    diffusion_coefficient=2e-9
)
# Output: 1.5079644737231005e-14
```

```py title="Array input"
import particula as par
par.dynamics.get_first_order_mass_transport_k(
    particle_radius=np.array([1e-6, 2e-6]),
    vapor_transition=np.array([0.6, 0.6]),
    diffusion_coefficient=2e-9
)
# Output: array([1.50796447e-14, 6.03185789e-14])
```

> **References**
>
> - Aerosol Modeling: Chapter 2, Equation 2.49
> - Wikipedia contributors, "Mass diffusivity,"
>   https://en.wikipedia.org/wiki/Mass_diffusivity

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L42-L106)

## particula.dynamics.condensation.mass_transfer.get_latent_heat_energy_released

*function*

```python
def get_latent_heat_energy_released(mass_transfer: Union[float, NDArray[np.float64]], latent_heat: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate latent heat energy released during phase change.

This diagnostic function converts the mass transferred in a time step into
energy released to (or absorbed from) the gas phase:

- Q = dm × L
    - Q : Latent heat energy [J].
    - dm : Mass transferred per step [kg].
    - L : Latent heat of vaporization [J/kg].

Positive mass transfer (condensation) yields Q > 0, while negative mass
transfer (evaporation) yields Q < 0.

**Parameters**

- `mass_transfer` (`Union[float, NDArray[np.float64]]`) — Mass transferred per step [kg].
- `latent_heat` (`Union[float, NDArray[np.float64]]`) — Latent heat of vaporization [J/kg].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Latent heat energy released or absorbed [J].

**Examples**

```py title="Condensation releases heat"
import particula as par

par.dynamics.get_latent_heat_energy_released(
    mass_transfer=1.0e-15,
    latent_heat=2.454e6,
)
# Output: 2.454e-09
```

```py title="Evaporation absorbs heat"
import particula as par

par.dynamics.get_latent_heat_energy_released(
    mass_transfer=-1.0e-15,
    latent_heat=2.454e6,
)
# Output: -2.454e-09
```

**Raises**

- `ValueError` — If mass_transfer is non-finite or latent_heat is negative.

> **Notes**
>
> Large magnitudes may overflow following NumPy's float64 behavior.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L334-L397)

## particula.dynamics.condensation.mass_transfer.get_mass_transfer

*function*

```python
def get_mass_transfer(mass_rate: NDArray[np.float64], time_step: float, gas_mass: NDArray[np.float64], particle_mass: NDArray[np.float64], particle_concentration: NDArray[np.float64]) -> NDArray[np.float64]
```

Route mass transfer calculation to single or multiple-species routines.

Depending on whether gas_mass represents one or multiple species, this
function calls either calculate_mass_transfer_single_species or
calculate_mass_transfer_multiple_species. The primary calculation
involves:

- mass_to_change = mass_rate × time_step × particle_concentration

**Parameters**

- `mass_rate` (`NDArray[np.float64]`) — The rate of mass transfer per particle [kg/s].
- `time_step` (`float`) — The time step for the mass transfer calculation [s].
- `gas_mass` (`NDArray[np.float64]`) — The available mass of gas species [kg].
- `particle_mass` (`NDArray[np.float64]`) — The mass of each particle [kg].
- `particle_concentration` (`NDArray[np.float64]`) — The concentration of particles [#/m³].

**Returns**

- (`NDArray[np.float64]`) — The mass transferred (array with the same shape as particle_mass).

**Examples**

```py title="Single species input"
import particula as par
par.dynamics.get_mass_transfer(
    mass_rate=np.array([0.1, 0.5]),
    time_step=10,
    gas_mass=np.array([0.5]),
    particle_mass=np.array([1.0, 50]),
    particle_concentration=np.array([1, 0.5])
)
```

```py title="Multiple species input"
import particula as par
par.dynamics.get_mass_transfer(
    mass_rate=np.array([[0.1, 0.05, 0.03], [0.2, 0.15, 0.07]]),
    time_step=10,
    gas_mass=np.array([1.0, 0.8, 0.5]),
    particle_mass=np.array([[1.0, 0.9, 0.8], [1.2, 1.0, 0.7]]),
    particle_concentration=np.array([5, 4])
)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L460-L533)

## particula.dynamics.condensation.mass_transfer.get_mass_transfer_of_multiple_species

*function*

```python
def get_mass_transfer_of_multiple_species(mass_rate: NDArray[np.float64], time_step: float, gas_mass: NDArray[np.float64], particle_mass: NDArray[np.float64], particle_concentration: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate mass transfer for multiple gas species.

Here, gas_mass has multiple elements (each species). For each species,
this function calculates mass_to_change for all particle bins:

- mass_to_change = mass_rate × time_step × particle_concentration

Then it limits or scales that mass based on available gas mass and
particle mass in each species bin.

1. Computes the mass change each particle *would* take during `time_step`.
2. Scales condensation so the **column sum** never exceeds `gas_mass`.
3. Scales evaporation so the **column sum** never exceeds the particle
   inventory of that species.
4. Clips the result so no individual bin evaporates more mass than it owns.

**Parameters**

- `mass_rate` (`NDArray[np.float64]`) — The mass transfer rate per particle for each gas species [kg/s].
- `time_step` (`float`) — The time step [s].
- `gas_mass` (`NDArray[np.float64]`) — The available mass of each gas species [kg].
- `particle_mass` (`NDArray[np.float64]`) — The mass of each particle for each gas species [kg].
- `particle_concentration` (`NDArray[np.float64]`) — The concentration of particles [#/m³].

**Returns**

- (`NDArray[np.float64]`) — The mass transferred for multiple gas species, matching the shape of (particle_mass).

**Examples**

```py title="Multiple species input"
import particula as par
par.dynamics.get_mass_transfer_of_multiple_species(
    mass_rate=np.array([[0.1, 0.05, 0.03], [0.2, 0.15, 0.07]]),
    time_step=10,
    gas_mass=np.array([1.0, 0.8, 0.5]),
    particle_mass=np.array([[1.0, 0.9, 0.8], [1.2, 1.0, 0.7]]),
    particle_concentration=np.array([5, 4])
)
# Output: array([...])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L605-L677)

## particula.dynamics.condensation.mass_transfer.get_mass_transfer_of_single_species

*function*

```python
def get_mass_transfer_of_single_species(mass_rate: NDArray[np.float64], time_step: float, gas_mass: NDArray[np.float64], particle_mass: NDArray[np.float64], particle_concentration: NDArray[np.float64]) -> NDArray[np.float64]
```

Calculate mass transfer for a single gas species.

This function assumes gas_mass has a size of 1 (single species).
It first computes the total mass_to_change per particle:

- mass_to_change = mass_rate × time_step × particle_concentration

Then it scales or limits that mass based on available gas mass and
particle mass.

**Parameters**

- `mass_rate` (`NDArray[np.float64]`) — Mass transfer rate per particle [kg/s].
- `time_step` (`float`) — The time step [s].
- `gas_mass` (`NDArray[np.float64]`) — Total available mass of the gas species [kg].
- `particle_mass` (`NDArray[np.float64]`) — The mass of each particle [kg].
- `particle_concentration` (`NDArray[np.float64]`) — Particle concentration [#/m³].

**Returns**

- (`NDArray[np.float64]`) — The amount of mass transferred for the single gas species, shaped like particle_mass.

**Examples**

```py title="Single species input"
import particula as par
par.dynamics.get_mass_transfer_of_single_species(
    mass_rate=np.array([0.1, 0.5]),
    time_step=10,
    gas_mass=np.array([0.5]),
    particle_mass=np.array([1.0, 50]),
    particle_concentration=np.array([1, 0.5])
)
# Output: array([...])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L536-L601)

## particula.dynamics.condensation.mass_transfer.get_mass_transfer_rate

*function*

```python
def get_mass_transfer_rate(pressure_delta: Union[float, NDArray[np.float64]], first_order_mass_transport: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the mass transfer rate for a particle.

This function calculates the mass transfer rate dm/dt, leveraging the
difference in partial pressure (pressure_delta) and the first-order
mass transport coefficient (K). The equation is:

- dm/dt = (K × Δp × M) / (R × T)
    - dm/dt : Mass transfer rate [kg/s].
    - K : First-order mass transport coefficient [m³/s].
    - Δp : Partial pressure difference [Pa].
    - M : Molar mass [kg/mol].
    - R : Universal gas constant [J/(mol·K)].
    - T : Temperature [K].

**Parameters**

- `pressure_delta` (`Union[float, NDArray[np.float64]]`) — The difference in partial pressure [Pa].
- `first_order_mass_transport` (`Union[float, NDArray[np.float64]]`) — The mass transport coefficient [m³/s].
- `temperature` (`Union[float, NDArray[np.float64]]`) — The temperature [K].
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — The molar mass [kg/mol].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The mass transfer rate [kg/s].

**Examples**

```py title="Single value input"
import particula as par
par.dynamics.mass_transfer_rate(
    pressure_delta=10.0,
    first_order_mass_transport=1e-17,
    temperature=300.0,
    molar_mass=0.02897
)
# Output: 1.16143004e-21
```

```py title="Array input"
import particula as par
par.dynamics.mass_transfer_rate(
    pressure_delta=np.array([10.0, 15.0]),
    first_order_mass_transport=np.array([1e-17, 2e-17]),
    temperature=300.0,
    molar_mass=0.02897
)
# Output: array([1.16143004e-21, 3.48429013e-21])
```

> **References**
>
> - Aerosol Modeling: Chapter 2, Equation 2.41
> - Seinfeld and Pandis, "Atmospheric Chemistry and Physics,"
>     Equation 13.3

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L109-L180)

## particula.dynamics.condensation.mass_transfer.get_mass_transfer_rate_latent_heat

*function*

```python
def get_mass_transfer_rate_latent_heat(pressure_delta: Union[float, NDArray[np.float64]], first_order_mass_transport: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]], latent_heat: Union[float, NDArray[np.float64]], thermal_conductivity: Union[float, NDArray[np.float64]], vapor_pressure_surface: Union[float, NDArray[np.float64]], diffusion_coefficient: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate non-isothermal mass transfer rate with latent heat.

Applies the thermal resistance factor to the isothermal mass transfer
rate. When latent heat is zero, the correction equals unity and the
result matches get_mass_transfer_rate.

**Parameters**

- `pressure_delta` (`Union[float, NDArray[np.float64]]`) — Difference in partial pressure [Pa].
- `first_order_mass_transport` (`Union[float, NDArray[np.float64]]`) — Mass transport coefficient K [m³/s].
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature T [K].
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass M [kg/mol].
- `latent_heat` (`Union[float, NDArray[np.float64]]`) — Latent heat of vaporization L [J/kg].
- `thermal_conductivity` (`Union[float, NDArray[np.float64]]`) — Gas thermal conductivity kappa [W/(m·K)].
- `vapor_pressure_surface` (`Union[float, NDArray[np.float64]]`) — Equilibrium vapor pressure at the surface [Pa].
- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) — Vapor diffusion coefficient D [m²/s].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Non-isothermal mass transfer rate [kg/s], matching the broadcasted input shape.

**Raises**

- `ValueError` — If any validated inputs violate positive/nonnegative
constraints.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L259-L331)

## particula.dynamics.condensation.mass_transfer.get_radius_transfer_rate

*function*

```python
def get_radius_transfer_rate(mass_rate: Union[float, NDArray[np.float64]], particle_radius: Union[float, NDArray[np.float64]], density: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Convert mass rate to radius growth/evaporation rate.

This function converts the mass transfer rate (dm/dt) into a radius
change rate (dr/dt). The equation is:

- dr/dt = (1 / 4πr²ρ) × dm/dt
    - dr/dt : Radius change rate [m/s].
    - r : Particle radius [m].
    - ρ : Particle density [kg/m³].
    - dm/dt : Mass change rate [kg/s].

**Parameters**

- `mass_rate` (`Union[float, NDArray[np.float64]]`) — The mass transfer rate [kg/s].
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle [m].
- `density` (`Union[float, NDArray[np.float64]]`) — The density of the particle [kg/m³].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The radius growth (or evaporation) rate [m/s].

**Examples**

```py title="Single value input"
import particula as par
par.dynamics.radius_transfer_rate(
    mass_rate=1e-21,
    particle_radius=1e-6,
    density=1000
)
# Output: 7.95774715e-14
```

```py title="Array input"
import particula as par
par.dynamics.radius_transfer_rate(
    mass_rate=np.array([1e-21, 2e-21]),
    particle_radius=np.array([1e-6, 2e-6]),
    density=1000
)
# Output: array([7.95774715e-14, 1.98943679e-14])
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L400-L457)

## particula.dynamics.condensation.mass_transfer.get_thermal_resistance_factor

*function*

```python
def get_thermal_resistance_factor(diffusion_coefficient: Union[float, NDArray[np.float64]], latent_heat: Union[float, NDArray[np.float64]], vapor_pressure_surface: Union[float, NDArray[np.float64]], thermal_conductivity: Union[float, NDArray[np.float64]], temperature: Union[float, NDArray[np.float64]], molar_mass: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]
```

Calculate the thermal resistance factor for mass transfer.

Uses the non-isothermal correction from Topping & Bane (2022),
Equation 2.36. Inputs include diffusion coefficient D [m²/s],
latent heat L [J/kg], vapor pressure at the surface p_surf [Pa],
thermal conductivity kappa [W/(m·K)], temperature T [K], and
molar mass M [kg/mol]. When latent_heat is zero, the factor
reduces to r_specific * temperature.

**Parameters**

- `diffusion_coefficient` (`Union[float, NDArray[np.float64]]`) — Vapor diffusion coefficient D [m²/s].
- `latent_heat` (`Union[float, NDArray[np.float64]]`) — Latent heat of vaporization L [J/kg].
- `vapor_pressure_surface` (`Union[float, NDArray[np.float64]]`) — Equilibrium vapor pressure at the surface [Pa].
- `thermal_conductivity` (`Union[float, NDArray[np.float64]]`) — Gas thermal conductivity kappa [W/(m·K)].
- `temperature` (`Union[float, NDArray[np.float64]]`) — Temperature T [K].
- `molar_mass` (`Union[float, NDArray[np.float64]]`) — Molar mass M [kg/mol].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Thermal resistance factor for non-isothermal mass transfer [J/kg],
- (`Union[float, NDArray[np.float64]]`) — matching the broadcasted input shape.

**Raises**

- `ValueError` — If any validated inputs violate positive/nonnegative
constraints.
- `ValueError` — If the thermal resistance factor is non-positive.

> **References**
>
> - Topping, D., & Bane, M. (2022). Introduction to Aerosol
>   Modelling. Equation 2.36.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/condensation/mass_transfer.py#L183-L256)
