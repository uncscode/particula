# particula.particles.properties.aerodynamic_mobility_module

*module*

Module for aerodynamic mobility of a particle in a fluid.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_mobility_module.py#L1-L1)

## particula.particles.properties.aerodynamic_mobility_module.logger

*attribute* · *module attribute*

```python
logger = logging.getLogger('particula')
```

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_mobility_module.py#L11-L11)

## particula.particles.properties.aerodynamic_mobility_module.get_aerodynamic_mobility

*function*

```python
def get_aerodynamic_mobility(particle_radius: Union[float, NDArray[np.float64]], slip_correction_factor: Union[float, NDArray[np.float64]], dynamic_viscosity: float) -> Union[float, NDArray[np.float64]]
```

Calculate the aerodynamic mobility of a particle using classical fluid
mechanics.

The aerodynamic mobility (B) can be determined by:

- B = C / (6πμr)
    - B is the aerodynamic mobility (m²/s).
    - C is the slip correction factor (dimensionless).
    - μ is the dynamic viscosity of the fluid (Pa·s).
    - r is the radius of the particle (m).

**Parameters**

- `particle_radius` (`Union[float, NDArray[np.float64]]`) — The radius of the particle in meters.
- `slip_correction_factor` (`Union[float, NDArray[np.float64]]`) — Slip correction factor (dimensionless).
- `dynamic_viscosity` (`float`) — Dynamic viscosity of the fluid in Pa·s.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — The particle aerodynamic mobility in m²/s.

**Examples**

``` py title="Example"
import particula as par
par.particles.get_particle_aerodynamic_mobility(
    particle_radius=0.00005,
    slip_correction_factor=1.1,
    dynamic_viscosity=0.0000181
)

> **References**
>
> - Wikipedia contributors, "Stokes' Law," Wikipedia,
> https://en.wikipedia.org/wiki/Stokes%27_law.

[View source](https://github.com/uncscode/particula/blob/main/particula/particles/properties/aerodynamic_mobility_module.py#L14-L60)
