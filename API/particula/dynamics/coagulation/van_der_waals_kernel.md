# particula.dynamics.coagulation.van_der_waals_kernel

*module*

Thermal capture by nonretarded dispersion and polarizable spheres.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L1-L1)

## particula.dynamics.coagulation.van_der_waals_kernel.COULOMB_LENGTH_T

*attribute* · *module attribute*

```python
COULOMB_LENGTH_T = ELEMENTARY_CHARGE_VALUE ** 2 / 4 * np.pi * VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L35-L37)

## particula.dynamics.coagulation.van_der_waals_kernel.DISTANCES

*attribute* · *module attribute*

```python
DISTANCES = np.sin(np.linspace(0, np.pi / 2, RADIAL_INTERVALS + 1)) ** 2
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L34-L34)

## particula.dynamics.coagulation.van_der_waals_kernel.LOG_RATIO_STEP

*attribute* · *module attribute*

```python
LOG_RATIO_STEP = np.log(MAX_RADIUS_RATIO) / RATIO_INTERVALS
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L33-L33)

## particula.dynamics.coagulation.van_der_waals_kernel.MAX_RADIUS_RATIO

*attribute* · *module attribute*

```python
MAX_RADIUS_RATIO = 32.0
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L32-L32)

## particula.dynamics.coagulation.van_der_waals_kernel.RADIAL_INTERVALS

*attribute* · *module attribute*

```python
RADIAL_INTERVALS = 128
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L30-L30)

## particula.dynamics.coagulation.van_der_waals_kernel.RATIO_INTERVALS

*attribute* · *module attribute*

```python
RATIO_INTERVALS = 128
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L31-L31)

## particula.dynamics.coagulation.van_der_waals_kernel.TABLE_VERSION

*attribute* · *module attribute*

```python
TABLE_VERSION = 1
```

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L38-L38)

## particula.dynamics.coagulation.van_der_waals_kernel._ballistic_enhancement

*function*

```python
def _ballistic_enhancement(energy, barrier, distances)
```

Integrate the lower envelope of all angular momentum constraints.

At kinetic energy t in thermal units, each radius imposes
t*b²/R² <= (t-U/kT)/x². The active constraint moves outward in x
order as t increases. Integrate this piecewise linear minimum against
exp(-t) analytically, starting above the radial repulsion barrier.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L65-L95)

## particula.dynamics.coagulation.van_der_waals_kernel.dielectric_row

*function* · *cached*

```python
def dielectric_row(relative_permittivity, index)
```

Tabulate a material's geometrical response at one radius ratio.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L49-L62)

## particula.dynamics.coagulation.van_der_waals_kernel.get_van_der_waals_kernel_via_system_state

*function*

```python
def get_van_der_waals_kernel_via_system_state(particle_radius, particle_mass, particle_charge, temperature, pressure, hamaker_constant, relative_permittivity, *, exclude_self_pairs=False)
```

Return the symmetric thermal encounter matrix in m³/s.

**Parameters**

- `particle_radius` — Physical sphere radii [m], scalar or vector.
- `particle_mass` — Per-particle masses [kg], matching the radii.
- `particle_charge` — Signed numbers of elementary charges, matching radii.
- `temperature` — Gas and particle temperature [K].
- `pressure` — Gas pressure [Pa].
- `hamaker_constant` — Effective dispersion constant across air [J].
- `relative_permittivity` — Relative static permittivity of both particles.
- `exclude_self_pairs` (default: `False`) — For actual resolved slots, omit a slot paired with itself. Distinct slots with equal properties still interact.

**Returns**

- Square kernel array for compact polarizable spheres with unit sticking.

> **References**
>
> Ouyang et al. (2012), doi:10.1063/1.4742064.
> Bichoutskaia et al. (2010), doi:10.1063/1.3457157.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L153-L249)

## particula.dynamics.coagulation.van_der_waals_kernel.pair_enhancements

*function*

```python
def pair_enhancements(radius1, radius2, z1, z2, temperature, hamaker, epsilon)
```

Return continuum and free-molecular rate ratios for one pair.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L98-L150)

## particula.dynamics.coagulation.van_der_waals_kernel.validate_material

*function*

```python
def validate_material(hamaker_constant, relative_permittivity)
```

Validate material scalars for nonretarded dielectric-sphere capture.

[View source](https://github.com/uncscode/particula/blob/main/particula/dynamics/coagulation/van_der_waals_kernel.py#L41-L46)
