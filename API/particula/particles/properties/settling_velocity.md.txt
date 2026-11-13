# particula.particles.properties.settling_velocity

*module*

Particle settling velocity in a fluid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L1-L1)

## particula.particles.properties.settling_velocity._drag_coefficient

*function*

```python
def _drag_coefficient(reynolds_number: float) -> float
```

Return drag coefficient c_d given a Reynolds number Re.

**Parameters**

- `reynolds_number` (`float`) — Reynolds number [-].

**Returns**

- (`float`) — Drag coefficient c_d [-].

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L399-L415)

## particula.particles.properties.settling_velocity._velocity_mismatch

*function*

```python
def _velocity_mismatch(velocity: float, radius: float, rho_p: float, fluid_density: float, kinematic_viscosity: float, gravitational_acceleration: float) -> float
```

Calculate the mismatch between predicted and actual velocities.

**Parameters**

- `velocity` (`float`) — Current estimate of particle velocity [m/s].
- `radius` (`float`) — Particle radius [m].
- `rho_p` (`float`) — Particle density [kg/m³].
- `fluid_density` (`float`) — Fluid density [kg/m³].
- `kinematic_viscosity` (`float`) — Kinematic viscosity of the fluid [m²/s].
- `gravitational_acceleration` (`float`) — Gravitational acceleration [m/s²].

**Returns**

- (`float`) — Squared difference between predicted and actual velocities.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L419-L452)

## particula.particles.properties.settling_velocity.get_particle_settling_velocity

*function*

```python
def get_particle_settling_velocity(particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], slip_correction_factor: Union[float, NDArray[np.float64]], dynamic_viscosity: float, gravitational_acceleration: float = STANDARD_GRAVITY, fluid_density: float = 0.0) -> Union[float, NDArray[np.float64]]
```

Calculate particle settling velocity using Stokes' law.

The settling velocity (vₛ) is given by the equation:

- vₛ = (2 × r² × (ρₚ − ρ_f) × g × C_c) / (9 × μ)
    - vₛ : Settling velocity in m/s
    - r : Particle radius in m
    - ρₚ : Particle density in kg/m³
    - ρ_f : Fluid density in kg/m³
    - g : Gravitational acceleration in m/s²
    - C_c : Cunningham slip correction factor (dimensionless)
    - μ : Dynamic viscosity in Pa·s

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle in meters.
- `particle_density` (`Union[float, NDArray[np.float64]]`) — The density of the particle in kg/m³.
- `slip_correction_factor` (`Union[float, NDArray[np.float64]]`) — Account for non-continuum effects (dimensionless).
- `dynamic_viscosity` (`float`) — Dynamic viscosity of the fluid in Pa·s.
- `gravitational_acceleration` (`float`) (default: `STANDARD_GRAVITY`) — Gravitational acceleration in m/s².
- `fluid_density` (`float`) (default: `0.0`) — The fluid density in kg/m³. Defaults to 0.0.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Settling velocity of the particle in m/s.

**Examples**

```py title="Array Input Example"
import numpy as np
import particula as par
par.particles.particle_settling_velocity(
    particle_radius=np.array([1e-6, 1e-5, 1e-4]),
    particle_density=np.array([1000, 2000, 3000]),
    slip_correction_factor=np.array([1, 1, 1]),
    dynamic_viscosity=1.0e-3
)
# Output: array([...])
```

> **References**
>
> - "Stokes' Law," Wikipedia,
>   https://en.wikipedia.org/wiki/Stokes%27_law

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L31-L99)

## particula.particles.properties.settling_velocity.get_particle_settling_velocity_via_inertia

*function*

```python
def get_particle_settling_velocity_via_inertia(particle_inertia_time: Union[float, NDArray[np.float64]], particle_radius: Union[float, NDArray[np.float64]], relative_velocity: Union[float, NDArray[np.float64]], slip_correction_factor: Union[float, NDArray[np.float64]], gravitational_acceleration: float, kinematic_viscosity: float) -> Union[float, NDArray[np.float64]]
```

Calculate gravitational settling velocity using particle inertia time.

The settling velocity (vₛ) is determined by:

- vₛ = (g × τₚ × C_c) / f(Reₚ)
    - g is gravitational acceleration (m/s²).
    - τₚ is particle inertia time (s).
    - C_c is the Cunningham slip correction factor (dimensionless).
    - f(Reₚ) is the drag correction factor, 1 + 0.15 × Reₚ^0.687.
    - Reₚ is particle Reynolds number (dimensionless).

**Parameters**

- `particle_inertia_time` (`Union[float, NDArray[np.float64]]`) — Particle inertia time in seconds (s).
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters (m).
- `relative_velocity` (`Union[float, NDArray[np.float64]]`) — Relative velocity between particle and fluid (m/s).
- `slip_correction_factor` (`Union[float, NDArray[np.float64]]`) — Cunningham slip correction factor (dimensionless).
- `gravitational_acceleration` (`float`) — Gravitational acceleration (m/s²).
- `kinematic_viscosity` (`float`) — Kinematic viscosity of the fluid (m²/s).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Particle settling velocity in m/s.

**Examples**

```py title="Example"
import particula as par
par.particles.get_particle_settling_velocity_via_inertia(
    particle_inertia_time=0.002,
    particle_radius=1.0e-6,
    relative_velocity=0.1,
    slip_correction_factor=1.05,
    gravitational_acceleration=9.81,
    kinematic_viscosity=1.5e-5
)
# Output: ...
```

> **References**
>
> - Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008).
>   "Effects of turbulence on the geometric collision rate of
>   sedimenting droplets. Part 1. Results from direct numerical
>   simulation." New Journal of Physics, 10.
>   https://doi.org/10.1088/1367-2630/10/7/075015

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L102-L174)

## particula.particles.properties.settling_velocity.get_particle_settling_velocity_via_system_state

*function*

```python
def get_particle_settling_velocity_via_system_state(particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Compute the particle settling velocity based on system state parameters.

This function calculates the dynamic viscosity from temperature, the mean
free path from the same system state, and the Knudsen number of the
particle, then applies the slip correction factor. Finally, it returns
the settling velocity from Stokes' law with slip correction.

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters (m).
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density in kg/m³.
- `temperature` (`float`) — Temperature of the system in Kelvin (K).
- `pressure` (`float`) — Pressure of the system in Pascals (Pa).

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Settling velocity of the particle in m/s.

**Examples**

```py title="System State Example"
import particula as par
par.particles.particle_settling_velocity_via_system_state(
    particle_radius=1e-6,
    particle_density=1200,
    temperature=298.15,
    pressure=101325
)
# Output: ...
```

> **References**
>
> - Gas viscosity property estimation:
>   https://en.wikipedia.org/wiki/Viscosity#Gases
> - Slip correction and Knudsen number relations from:
>   Seinfeld, J. H., & Pandis, S. N. (2016). Atmospheric
>   Chemistry and Physics. Wiley-Interscience.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L177-L244)

## particula.particles.properties.settling_velocity.get_particle_settling_velocity_with_drag

*function*

```python
def get_particle_settling_velocity_with_drag(particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], fluid_density: float, dynamic_viscosity: float, slip_correction_factor: Union[float, NDArray[np.float64]], gravitational_acceleration: float = STANDARD_GRAVITY, re_threshold: float = 0.1, tol: float = 1e-06, max_iter: int = 100) -> Union[float, NDArray[np.float64]]
```

Calculate terminal settling velocity with full drag model.

For low Reynolds numbers (Re < re_threshold), the Stokes settling velocity
(with slip correction) is used:

- vₛ(Stokes) = (2/9) × [r² × (ρₚ − ρ_f) × g × C_c] / μ

For higher Reynolds numbers, a force-balance approach is solved
numerically, using a variable drag coefficient (c_d).

- vₛ = fminbound(mismatch, 0, v_upper)
    - mismatch = (v_pred - v)²
    - v_pred = sqrt((8 × r × (ρₚ - ρ_f) × g) / (3 × ρ_f × c_d))

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius (m).
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density (kg/m³).
- `fluid_density` (`float`) — Fluid density (kg/m³).
- `dynamic_viscosity` (`float`) — Fluid dynamic viscosity (Pa·s).
- `slip_correction_factor` (`Union[float, NDArray[np.float64]]`) — Slip correction factor, dimensionless.
- `gravitational_acceleration` (`float`) (default: `STANDARD_GRAVITY`) — Gravitational acceleration (m/s²), default is 9.80665.
- `re_threshold` (`float`) (default: `0.1`) — Reynolds number threshold (dimensionless), default is 0.1.
- `tol` (`float`) (default: `1e-06`) — Numeric tolerance for solver (dimensionless), default is 1e-6.
- `max_iter` (`int`) (default: `100`) — Maximum function evaluations in numeric solver, default is 100.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Terminal settling velocity (m/s). Scalar or NDArray.

**Examples**

```py title="Example"
import numpy as np
import particula as par
r_array = np.array([1e-6, 5e-5, 2e-4])
rho_array = np.array([1500, 2000, 1850])
par.particles.get_particle_settling_velocity_with_drag(
    particle_radius=r_array,
    particle_density=rho_array,
    fluid_density=1.225,
    dynamic_viscosity=1.8e-5,
    slip_correction_factor=np.array([1.0, 0.95, 1.1])
)
# Output: array([...])
```

> **References**
>
> - "Drag Coefficient," Wikipedia,
>   https://en.wikipedia.org/wiki/Drag_coefficient
> - Seinfeld, J. H., & Pandis, S. N. (2016). *Atmospheric Chemistry
>   and Physics*, 3rd ed., John Wiley & Sons.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/settling_velocity.py#L247-L396)
