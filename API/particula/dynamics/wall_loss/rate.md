# particula.dynamics.wall_loss.rate

*module*

Calculate wall loss rates for particles in various geometries.

Supports neutral spherical and rectangular rate helpers plus charged wall
loss with image-charge enhancement and optional electric-field drift.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/rate.py#L1-L1)

## particula.dynamics.wall_loss.rate.get_charged_wall_loss_rate

*function*

```python
def get_charged_wall_loss_rate(wall_eddy_diffusivity: float, particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], particle_concentration: Union[float, NDArray[np.float64]], particle_charge: Union[float, NDArray[np.float64]], temperature: float, pressure: float, chamber_geometry: str, chamber_radius: Union[float, None] = None, chamber_dimensions: Union[Tuple[float, float, float], None] = None, wall_potential: float = 0.0, wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0) -> Union[float, NDArray[np.float64]]
```

Calculate charged wall loss rate for spherical or rectangular chambers.

Combines neutral deposition with image-charge enhancement and optional
electric-field drift. Image-charge effects apply when particles carry
charge even if ``wall_potential`` is zero; drift is applied only when
``wall_electric_field`` is non-zero. Reduces to the neutral rate when
both charge and electric field are zero.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity in s⁻¹.
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius in meters.
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density in kg/m³.
- `particle_concentration` (`Union[float, NDArray[np.float64]]`) — Particle concentration in #/m³.
- `particle_charge` (`Union[float, NDArray[np.float64]]`) — Particle charge in elementary charge units.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.
- `chamber_geometry` (`str`) — Geometry name, ``"spherical"`` or ``"rectangular"``.
- `chamber_radius` (`Union[float, None]`) (default: `None`) — Chamber radius in meters for spherical geometry.
- `chamber_dimensions` (`Union[Tuple[float, float, float], None]`) (default: `None`) — ``(length, width, height)`` in meters for rectangular geometry.
- `wall_potential` (`float`) (default: `0.0`) — Wall potential in volts. Zero still enables image-charge effects when charge is non-zero.
- `wall_electric_field` (`Union[float, Tuple[float, float, float]]`) (default: `0.0`) — Electric field magnitude in V/m (scalar for spherical or tuple for rectangular). Zero disables drift.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Charged wall loss rate in #/m³·s as scalar or array matching
- (`Union[float, NDArray[np.float64]]`) — ``particle_concentration``.

**Raises**

- `ValueError` — If geometry arguments are inconsistent or not finite.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/rate.py#L142-L208)

## particula.dynamics.wall_loss.rate.get_rectangle_wall_loss_rate

*function*

```python
def get_rectangle_wall_loss_rate(wall_eddy_diffusivity: float, particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], particle_concentration: Union[float, NDArray[np.float64]], temperature: float, pressure: float, chamber_dimensions: Tuple[float, float, float]) -> Union[float, NDArray[np.float64]]
```

Calculate the wall loss rate for a rectangular chamber.

The loss rate is computed as ``L = -k c`` where ``k`` is the wall
loss coefficient [1/s] and ``c`` is the particle concentration
[#/m³]. The coefficient is obtained from the rectangular wall loss
coefficient computed via
:func:`get_rectangle_wall_loss_coefficient_via_system_state`.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [s⁻¹].
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius [m].
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density [kg/m³].
- `particle_concentration` (`Union[float, NDArray[np.float64]]`) — Particle concentration [#/m³].
- `temperature` (`float`) — System temperature [K].
- `pressure` (`float`) — System pressure [Pa].
- `chamber_dimensions` (`Tuple[float, float, float]`) — Chamber dimensions ``(length, width, height)`` [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m³·s] as a scalar or array.

**Examples**

```pycon
>>> import particula as par
>>> loss_rate = par.dynamics.wall_loss.get_rectangle_wall_loss_rate(
...     wall_eddy_diffusivity=1e-4,
...     particle_radius=5e-8,
...     particle_density=1200.0,
...     particle_concentration=2e10,
...     temperature=300.0,
...     pressure=101325.0,
...     chamber_dimensions=(1.0, 0.5, 0.5),
... )
>>> loss_rate
```

> **References**
>
> Hinds, W. C. "Aerosol Technology." 2nd ed. John Wiley & Sons, 1999.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/rate.py#L83-L139)

## particula.dynamics.wall_loss.rate.get_spherical_wall_loss_rate

*function*

```python
def get_spherical_wall_loss_rate(wall_eddy_diffusivity: float, particle_radius: Union[float, NDArray[np.float64]], particle_density: Union[float, NDArray[np.float64]], particle_concentration: Union[float, NDArray[np.float64]], temperature: float, pressure: float, chamber_radius: float) -> Union[float, NDArray[np.float64]]
```

Calculate the wall loss rate for a spherical chamber.

The loss rate is computed as ``L = -k c`` where ``k`` is the wall
loss coefficient [1/s] and ``c`` is the particle concentration
[#/m³]. The coefficient is obtained from the spherical wall loss
coefficient computed via
:func:`get_spherical_wall_loss_coefficient_via_system_state`.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [s⁻¹].
- `particle_radius` (`Union[float, NDArray[np.float64]]`) — Particle radius [m].
- `particle_density` (`Union[float, NDArray[np.float64]]`) — Particle density [kg/m³].
- `particle_concentration` (`Union[float, NDArray[np.float64]]`) — Particle concentration [#/m³].
- `temperature` (`float`) — System temperature [K].
- `pressure` (`float`) — System pressure [Pa].
- `chamber_radius` (`float`) — Radius of the spherical chamber [m].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m³·s] as a scalar or array.

**Examples**

```pycon
>>> import particula as par
>>> rate = par.dynamics.wall_loss.get_spherical_wall_loss_rate(
...     wall_eddy_diffusivity=1e-3,
...     particle_radius=1e-7,
...     particle_density=1000.0,
...     particle_concentration=1e11,
...     temperature=298.0,
...     pressure=101325.0,
...     chamber_radius=0.5,
... )
>>> rate
```

> **References**
>
> Wikipedia contributors. "Aerosol dynamics." *Wikipedia*.
> https://en.wikipedia.org/wiki/Aerosol.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/rate.py#L22-L79)
