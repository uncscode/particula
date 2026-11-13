# particula.dynamics.wall_loss.wall_loss_strategies

*module*

Wall loss strategy abstractions and implementations.

Defines abstract and concrete strategies for modeling particle wall
loss processes in different chamber geometries. Strategies operate on
:class:`~particula.particles.particle_data.ParticleData`
objects with explicit distribution metadata. Implementations are
provided for spherical and rectangular chambers.

Strategies require explicitly tagged single-box ``ParticleData``.
Native PMF/PDF values are densities at unit volume; resolved values are counts
and rates use count/volume. Native steps preserve caller-owned arrays and stage
updates before publication. Facade inputs are no longer supported.

The wall loss rate is modeled as a first-order size dependent loss
process

.. math::

    L = -k (Dp) c,

where :math:`L` is the wall loss rate [#/m^3 s],
:math:`k` is the wall loss coefficient [1/s], and :math:`c` is the
particle number concentration [#/m^3].

> **References**
>
> Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and
> gravitational sedimentation of an aerosol in a vessel of arbitrary
> shape. Journal of Aerosol Science, 12(5), 405–415.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L1-L1)

## particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy

*class*

```python
class ChargedWallLossStrategy(WallLossStrategy)
```

Bases: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Wall loss strategy with electrostatic effects.

Extends neutral wall loss with image-charge enhancement, optional
electric-field drift, and diffusion modification for charged particles.
Behaves as the neutral strategy when both charge and field terms are zero
while still applying the image-charge-only enhancement when charge is
non-zero and ``wall_potential`` is zero.

> **References**
>
> McMurry, P. H., & Rader, D. J. (1985). Aerosol wall losses in
> electrically charged chambers. Aerosol Science and Technology, 4(3),
> 249–268.
> Lai, A. C. K., & Nazaroff, W. W. (2000). Modeling indoor particle
> deposition from turbulent flow onto smooth surfaces. J. Aerosol Sci.,
> 31(4), 463–476.
> Hinds, W. C. (1999). *Aerosol Technology*. Wiley.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L411-L812)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Union[Tuple[float, float, float], None]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L432-L432)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.chamber_geometry

*attribute* · *instance attribute*

```python
chamber_geometry: str = geometry
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L503-L503)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: Union[float, None]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L431-L431)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.wall_electric_field

*attribute* · *instance attribute*

```python
wall_electric_field: Union[float, Tuple[float, float, float]] = wall_electric_field
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L505-L505)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.wall_potential

*attribute* · *instance attribute*

```python
wall_potential: float = float(wall_potential)
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L504-L504)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = distribution_type
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L176-L176)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator: np.random.Generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L177-L177)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float = wall_eddy_diffusivity
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L175-L175)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy._geometry_scale

*property*

```python
_geometry_scale: float
```

Return characteristic geometry length for field scaling.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L508-L514)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.__init__

*method*

```python
def __init__(wall_eddy_diffusivity: float, chamber_geometry: str, chamber_radius: Union[float, None] = None, chamber_dimensions: Union[Tuple[float, float, float], None] = None, wall_potential: float = 0.0, wall_electric_field: Union[float, Tuple[float, float, float]] = 0.0, distribution_type: str = 'discrete') -> None
```

Initialize charged wall loss strategy.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_geometry` (`str`) — Geometry string ("spherical" or "rectangular").
- `chamber_radius` (`Union[float, None]`) (default: `None`) — Radius for spherical chambers [m].
- `chamber_dimensions` (`Union[Tuple[float, float, float], None]`) (default: `None`) — Dimensions (length, width, height) for rectangular chambers [m].
- `wall_potential` (`float`) (default: `0.0`) — Wall potential [V]; zero keeps image-charge term.
- `wall_electric_field` (`Union[float, Tuple[float, float, float]]`) (default: `0.0`) — Electric field magnitude [V/m] (scalar) or 3-vector for rectangular chambers. Zero disables drift term.
- `distribution_type` (`str`) (default: `'discrete'`) — Distribution type.

**Raises**

- `ValueError` — If geometry is invalid or required geometry parameters
are missing.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L436-L505)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy._combine_coefficients

*method*

```python
def _combine_coefficients(neutral: NDArray[np.float64], electrostatic_factor: NDArray[np.float64], drift_term: NDArray[np.float64]) -> NDArray[np.float64]
```

Combine neutral, electrostatic, and drift contributions.

**Parameters**

- `neutral` (`NDArray[np.float64]`) — Neutral wall loss coefficient in 1/s.
- `electrostatic_factor` (`NDArray[np.float64]`) — Multiplicative factor from image charge.
- `drift_term` (`NDArray[np.float64]`) — Additive drift term in 1/s.

**Returns**

- (`NDArray[np.float64]`) — Combined wall loss coefficient clipped to non-negative finite
- (`NDArray[np.float64]`) — values.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L650-L672)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy._drift_term

*method*

```python
def _drift_term(particle_radius: NDArray[np.float64], particle_charge: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute electric-field drift contribution.

Returns zero when charge is zero or when the resolved electric field
is zero.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_charge` (`NDArray[np.float64]`) — Particle charge in elementary charges.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`NDArray[np.float64]`) — Additive drift term in 1/s for the wall loss coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L612-L648)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy._electrostatic_factor

*method*

```python
def _electrostatic_factor(particle_radius: NDArray[np.float64], particle_charge: NDArray[np.float64], temperature: float) -> NDArray[np.float64]
```

Compute electrostatic enhancement factor.

Applies image-charge-derived enhancement based on particle charge.
Returns ones when charge is zero so the strategy reduces to neutral
even if ``wall_potential`` is zero.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_charge` (`NDArray[np.float64]`) — Particle charge in elementary charges.
- `temperature` (`float`) — Gas temperature in kelvin.

**Returns**

- (`NDArray[np.float64]`) — Multiplicative electrostatic factor for the wall loss coefficient.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L558-L591)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy._neutral_coefficient

*method*

```python
def _neutral_coefficient(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Return neutral wall loss coefficient for configured geometry.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_density` (`NDArray[np.float64]`) — Particle densities in kg/m³.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`NDArray[np.float64]`) — Neutral wall loss coefficient in 1/s for each particle size.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L516-L556)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy._resolve_electric_field

*method*

```python
def _resolve_electric_field() -> float
```

Return resolved electric field magnitude in V/m.

Combines the configured wall electric field (scalar or vector) with a
geometry-scaled potential-derived field; returns zero when both are
zero.

**Returns**

- (`float`) — Total electric field magnitude in volts per meter.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L593-L610)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.compute_coefficient_from_arrays

*method*

```python
def compute_coefficient_from_arrays(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], particle_charge: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute charged wall loss coefficient from explicit arrays.

Public method for computing coefficients with explicit charge arrays,
useful for rate helpers and direct calculations without cached state.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_density` (`NDArray[np.float64]`) — Particle densities in kg/m³.
- `particle_charge` (`NDArray[np.float64]`) — Particle charge in elementary charges.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient in 1/s for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L744-L788)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.loss_coefficient

*method*

```python
def loss_coefficient(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Compute charged wall loss coefficient from particle properties.

Applies image-charge enhancement even when ``wall_potential`` is zero
and adds drift when ``wall_electric_field`` is non-zero.
Reduces to the neutral coefficient when charge and field are zero and
supports particle-resolved active subsets.

**Parameters**

- `particle` (`ParticleData`) — Native particle data providing radius, density, and charge.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss coefficient in 1/s as a scalar or array.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L674-L696)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.loss_coefficient_for_particles

*method*

```python
def loss_coefficient_for_particles(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Compute charged wall loss coefficient for provided particle arrays.

This charge-free interface assumes neutral particles. For charged
material, use ``compute_coefficient_from_arrays`` with explicit charge
arrays. No earlier particle evaluation supplies implicit charge state.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii in meters.
- `particle_density` (`NDArray[np.float64]`) — Particle densities in kg/m³.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient in 1/s for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L698-L742)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Advance particle-resolved concentration with charged wall loss.

Computes survival probabilities using neutral, image-charge, and
optional drift contributions. Clamps probabilities to [0, 1] and
updates particle concentrations and distributions.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to update.
- `temperature` (`float`) — Gas temperature in kelvin.
- `pressure` (`float`) — Gas pressure in pascals.
- `time_step` (`float`) — Time step in seconds.

**Returns**

- (`ParticleData`) — Updated native particle data.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L790-L812)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate for the given state.

The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
size-dependent wall loss coefficient and ``c`` is the particle number
concentration.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m^3 s] (negative values indicate loss).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L220-L240)

### particula.dynamics.wall_loss.wall_loss_strategies.ChargedWallLossStrategy.rate

*method*

```python
def rate(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate as an array.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Array of wall loss rates [#/m^3 s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L242-L264)

## particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy

*class*

```python
class RectangularWallLossStrategy(WallLossStrategy)
```

Bases: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Wall loss strategy for rectangular (box) chambers.

Calculates particle wall deposition in rectangular chamber geometry
using turbulent diffusion and gravitational settling. Supports
discrete, continuous PDF, and particle-resolved distributions.

**Attributes**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_dimensions` (`Tuple[float, float, float]`) — Chamber dimensions (length, width, height) [m].
- `distribution_type` (`str`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

**Examples**

```pycon
>>> import particula as par
>>> particle = par.particles.create_lognormal_binned_particle_data(
...     distribution_type="discrete",
... )
>>> strategy = par.dynamics.RectangularWallLossStrategy(
...     wall_eddy_diffusivity=0.001,
...     chamber_dimensions=(1.0, 0.5, 0.5),
...     distribution_type="discrete",
... )
>>> rate = strategy.rate(
...     particle=particle,
...     temperature=298.0,
...     pressure=101325.0,
... )
>>> _ = strategy.step(
...     particle=particle,
...     temperature=298.0,
...     pressure=101325.0,
...     time_step=0.1,
... )
```

> **References**
>
> Crump, J. G., & Seinfeld, J. H. (1981). Turbulent deposition and
> gravitational sedimentation of an aerosol in a vessel of arbitrary
> shape. Journal of Aerosol Science, 12(5), 405–415.
> McMurry, P. H., & Rader, D. J. (1985). Aerosol wall losses in
> electrically charged chambers. Aerosol Science and Technology, 4(3),
> 249–268.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L815-L946)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.chamber_dimensions

*attribute* · *instance attribute*

```python
chamber_dimensions: Tuple[float, float, float] = (float(chamber_dimensions[0]), float(chamber_dimensions[1]), float(chamber_dimensions[2]))
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L892-L896)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = distribution_type
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L176-L176)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator: np.random.Generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L177-L177)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float = wall_eddy_diffusivity
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L175-L175)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.__init__

*method*

```python
def __init__(wall_eddy_diffusivity: float, chamber_dimensions: Tuple[float, float, float], distribution_type: str = 'discrete') -> None
```

Initialize rectangular wall loss strategy.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_dimensions` (`Tuple[float, float, float]`) — Chamber dimensions (length, width, height) in meters. All must be positive.
- `distribution_type` (`str`) (default: `'discrete'`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

**Raises**

- `ValueError` — If ``chamber_dimensions`` does not contain exactly
three positive values.
- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L861-L896)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.loss_coefficient

*method*

```python
def loss_coefficient(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Return the rectangular wall loss coefficient for the state.

**Parameters**

- `particle` (`ParticleData`) — Native particle data providing radius and density.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss coefficient [1/s] for each particle bin.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L898-L914)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.loss_coefficient_for_particles

*method*

```python
def loss_coefficient_for_particles(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Return wall loss coefficient for provided particle properties.

This method is used for particle-resolved simulations to evaluate
coefficients on active particles without reconstructing the full
representation.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Particle densities [kg/m^3].
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient [1/s] for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L916-L946)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate for the given state.

The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
size-dependent wall loss coefficient and ``c`` is the particle number
concentration.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m^3 s] (negative values indicate loss).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L220-L240)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.rate

*method*

```python
def rate(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate as an array.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Array of wall loss rates [#/m^3 s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L242-L264)

### particula.dynamics.wall_loss.wall_loss_strategies.RectangularWallLossStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Advance the native particle data by one wall loss step.

For ``"discrete"`` and ``"continuous_pdf"`` distributions, this
applies a deterministic first-order loss to the number
concentration in each bin. For ``"particle_resolved"``, it
applies a stochastic removal process using binomial random draws
based on survival probability.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to update.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].
- `time_step` (`float`) — Time step [s].

**Returns**

- (`ParticleData`) — Updated native particle data (same instance).

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L266-L293)

## particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy

*class*

```python
class SphericalWallLossStrategy(WallLossStrategy)
```

Bases: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Wall loss strategy for spherical chambers.

Calculates particle wall deposition in spherical chamber geometry
using turbulent diffusion and gravitational settling.

**Attributes**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_radius` (`float`) — Radius of the spherical chamber [m].
- `distribution_type` (`str`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

**Examples**

```pycon
>>> from particula.dynamics.wall_loss.wall_loss_strategies import (
...     SphericalWallLossStrategy,
... )
>>> strategy = SphericalWallLossStrategy(
...     wall_eddy_diffusivity=0.001,
...     chamber_radius=0.5,
...     distribution_type="discrete",
... )
>>> rate = strategy.rate(
...     particle=particle,
...     temperature=298.0,
...     pressure=101325.0,
... )
>>> particle = strategy.step(
...     particle=particle,
...     temperature=298.0,
...     pressure=101325.0,
...     time_step=1.0,
... )
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L296-L408)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.chamber_radius

*attribute* · *instance attribute*

```python
chamber_radius: float = chamber_radius
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L356-L356)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = distribution_type
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L176-L176)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator: np.random.Generator = np.random.default_rng()
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L177-L177)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float = wall_eddy_diffusivity
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L175-L175)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.__init__

*method*

```python
def __init__(wall_eddy_diffusivity: float, chamber_radius: float, distribution_type: str = 'discrete') -> None
```

Initialize spherical wall loss strategy.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `chamber_radius` (`float`) — Radius of the spherical chamber [m].
- `distribution_type` (`str`) (default: `'discrete'`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L332-L356)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.loss_coefficient

*method*

```python
def loss_coefficient(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Return the spherical wall loss coefficient for the given state.

The coefficient is computed from the system state using
:func:`get_spherical_wall_loss_coefficient_via_system_state`.

**Parameters**

- `particle` (`ParticleData`) — Native particle data providing radius and density.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss coefficient [1/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L358-L377)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.loss_coefficient_for_particles

*method*

```python
def loss_coefficient_for_particles(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Return the wall loss coefficient for given particle properties.

This method is used for particle-resolved simulations where we
need to calculate coefficients for a subset of active particles.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Particle densities [kg/m^3].
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient [1/s] for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L379-L408)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate for the given state.

The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
size-dependent wall loss coefficient and ``c`` is the particle number
concentration.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m^3 s] (negative values indicate loss).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L220-L240)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.rate

*method*

```python
def rate(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Return the wall loss rate as an array.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Array of wall loss rates [#/m^3 s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L242-L264)

### particula.dynamics.wall_loss.wall_loss_strategies.SphericalWallLossStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Inherited from: `particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy`

Advance the native particle data by one wall loss step.

For ``"discrete"`` and ``"continuous_pdf"`` distributions, this
applies a deterministic first-order loss to the number
concentration in each bin. For ``"particle_resolved"``, it
applies a stochastic removal process using binomial random draws
based on survival probability.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to update.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].
- `time_step` (`float`) — Time step [s].

**Returns**

- (`ParticleData`) — Updated native particle data (same instance).

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L266-L293)

## particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy

*class*

```python
class WallLossStrategy(ABC)
```

Bases: `ABC`

Abstract base class for wall loss strategies.

Wall loss strategies compute wall loss coefficients and associated
rates for different distribution types.

**Attributes**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `distribution_type` (`str`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").
- `random_generator` (`np.random.Generator`) — Random number generator for stochastic particle-resolved simulations.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L133-L293)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.distribution_type

*attribute* · *instance attribute*

```python
distribution_type: str = distribution_type
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L176-L176)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.random_generator

*attribute* · *instance attribute*

```python
random_generator: np.random.Generator = np.random.default_rng()
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L177-L177)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.wall_eddy_diffusivity

*attribute* · *instance attribute*

```python
wall_eddy_diffusivity: float = wall_eddy_diffusivity
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L175-L175)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.__init__

*method*

```python
def __init__(wall_eddy_diffusivity: float, distribution_type: str) -> None
```

Initialize the wall loss strategy.

**Parameters**

- `wall_eddy_diffusivity` (`float`) — Wall eddy diffusivity [1/s].
- `distribution_type` (`str`) — Distribution type ("discrete", "continuous_pdf", or "particle_resolved").

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L151-L177)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.loss_coefficient

*method* · *abstract*

```python
def loss_coefficient(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Return the wall loss coefficient for the given state.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss coefficient [1/s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L179-L195)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.loss_coefficient_for_particles

*method* · *abstract*

```python
def loss_coefficient_for_particles(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], temperature: float, pressure: float) -> NDArray[np.float64]
```

Return the wall loss coefficient for given particle properties.

This method is used for particle-resolved simulations where we
need to calculate coefficients for a subset of active particles.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Particle densities [kg/m^3].
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Wall loss coefficient [1/s] for each particle.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L197-L218)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.loss_rate

*method*

```python
def loss_rate(particle: ParticleData, temperature: float, pressure: float) -> Union[float, NDArray[np.float64]]
```

Return the wall loss rate for the given state.

The loss rate is computed as ``-k(Dp) * c`` where ``k(Dp)`` is the
size-dependent wall loss coefficient and ``c`` is the particle number
concentration.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`Union[float, NDArray[np.float64]]`) — Wall loss rate [#/m^3 s] (negative values indicate loss).

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L220-L240)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.rate

*method*

```python
def rate(particle: ParticleData, temperature: float, pressure: float) -> NDArray[np.float64]
```

Return the wall loss rate as an array.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to evaluate.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].

**Returns**

- (`NDArray[np.float64]`) — Array of wall loss rates [#/m^3 s].

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L242-L264)

### particula.dynamics.wall_loss.wall_loss_strategies.WallLossStrategy.step

*method*

```python
def step(particle: ParticleData, temperature: float, pressure: float, time_step: float) -> ParticleData
```

Advance the native particle data by one wall loss step.

For ``"discrete"`` and ``"continuous_pdf"`` distributions, this
applies a deterministic first-order loss to the number
concentration in each bin. For ``"particle_resolved"``, it
applies a stochastic removal process using binomial random draws
based on survival probability.

**Parameters**

- `particle` (`ParticleData`) — Native particle data to update.
- `temperature` (`float`) — Gas temperature [K].
- `pressure` (`float`) — Gas pressure [Pa].
- `time_step` (`float`) — Time step [s].

**Returns**

- (`ParticleData`) — Updated native particle data (same instance).

**Raises**

- `ValueError` — If ``distribution_type`` is not supported.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L266-L293)

## particula.dynamics.wall_loss.wall_loss_strategies.get_particle_resolved_wall_loss_step

*function*

```python
def get_particle_resolved_wall_loss_step(particle_radius: NDArray[np.float64], particle_density: NDArray[np.float64], concentration: NDArray[np.float64], loss_coefficient_func: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], time_step: float, random_generator: np.random.Generator) -> NDArray[np.float64]
```

Perform particle-resolved wall loss step with stochastic removal.

For particle-resolved simulations, each computational particle survives
with probability ``exp(-k * dt)``, where ``k`` is the size-dependent wall
loss coefficient. Binomial draws determine which particles are lost to the
walls.

**Parameters**

- `particle_radius` (`NDArray[np.float64]`) — Array of particle radii [m].
- `particle_density` (`NDArray[np.float64]`) — Array of particle densities [kg/m^3].
- `concentration` (`NDArray[np.float64]`) — Array of particle concentrations [#/m^3].
- `loss_coefficient_func` (`Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]`) — Callable returning wall loss coefficient for provided particle radius and density. Temperature, pressure, and geometry should be bound via closure.
- `time_step` (`float`) — Time step [s].
- `random_generator` (`np.random.Generator`) — Random number generator for stochastic draws.

**Returns**

- (`NDArray[np.float64]`) — Array of survival indicators (1.0 for survived, 0.0 for lost).

**Examples**

```pycon
>>> import numpy as np
>>> from particula.dynamics.wall_loss.wall_loss_strategies import (
...     get_particle_resolved_wall_loss_step,
... )
>>> radius = np.array([1e-7, 2e-7, 3e-7])
>>> density = np.array([1000.0, 1000.0, 1000.0])
>>> concentration = np.array([1e6, 1e6, 1e6])
>>> def coeff_func(r, d):
...     return 1e-4 * np.ones_like(r)
>>> rng = np.random.default_rng(42)
>>> survived = get_particle_resolved_wall_loss_step(
...     particle_radius=radius,
...     particle_density=density,
...     concentration=concentration,
...     loss_coefficient_func=coeff_func,
...     time_step=1.0,
...     random_generator=rng,
... )
>>> print(survived)
[1. 1. 1.]
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/wall_loss/wall_loss_strategies.py#L52-L130)
