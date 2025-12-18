# Wall Loss Builders and Factory

Create wall loss strategies with the new builders and factory API, including
unit-aware setters and distribution type controls. This guide shows how to:

- Chain `SphericalWallLossBuilder` and `RectangularWallLossBuilder` setters with
  automatic unit conversion.
- Build strategies directly or via `WallLossFactory.get_strategy` for
  "spherical" and "rectangular" types.
- Choose distribution types (defaults to `"discrete"`) and see validation
  errors for unsupported values.

## Prerequisites

- Python 3.9+
- `particula` installed (includes `pint` for unit conversion):

```bash
pip install particula
```

- Basic familiarity with NumPy arrays and Particula particle presets.

## Quick start: builder chaining with units

This example builds a spherical wall loss strategy while converting units
(`cm` → `m`, `1/hr` → `1/s`).

```python
import particula as par

# Sample particle distribution for sizing; keep it light-weight
particle = par.particles.PresetParticleRadiusBuilder().build()

spherical = (
    par.dynamics.SphericalWallLossBuilder()
    .set_wall_eddy_diffusivity(3.6, "1/hr")   # converts to 1.000e-03 1/s
    .set_chamber_radius(50.0, "cm")           # converts to 0.50 m
    # distribution_type defaults to "discrete"
    .build()
)

rate = spherical.rate(
    particle=particle,
    temperature=298.0,
    pressure=101325.0,
)

print("eddy diffusivity [1/s]", f"{spherical.wall_eddy_diffusivity:.3e}")
print("chamber radius [m]", f"{spherical.chamber_radius:.2f}")
print("distribution type", spherical.distribution_type)
print("rate shape", rate.shape)
print("rate[0] example", f"{rate[0]:.3e} 1/s")
```

**Expected output (approximate):**

```
eddy diffusivity [1/s] 1.000e-03
chamber radius [m] 0.50
distribution type discrete
rate shape (50,)
rate[0] example 1.5e-03 1/s
```

## Factory quick start: spherical and rectangular

`WallLossFactory` maps strategy names to builders and applies `set_parameters`
(including unit suffixes) plus `distribution_type`.

```python
import particula as par

factory = par.dynamics.WallLossFactory()

spherical = factory.get_strategy(
    "spherical",
    {
        "wall_eddy_diffusivity": 2.0,
        "wall_eddy_diffusivity_units": "1/min",  # → 3.333e-02 1/s
        "chamber_radius": 75.0,
        "chamber_radius_units": "cm",            # → 0.75 m
        # distribution_type omitted ⇒ defaults to "discrete"
    },
)

rectangular = factory.get_strategy(
    "rectangular",
    {
        "wall_eddy_diffusivity": 1.8,
        "wall_eddy_diffusivity_units": "1/hr",   # → 5.000e-04 1/s
        "chamber_dimensions": (200.0, 150.0, 120.0),
        "chamber_dimensions_units": "cm",        # → (2.0, 1.5, 1.2) m
        "distribution_type": "continuous_pdf",
    },
)

print("spherical diffusivity [1/s]", f"{spherical.wall_eddy_diffusivity:.3e}")
print("rectangular dims [m]", rectangular.chamber_dimensions)
print("rectangular distribution", rectangular.distribution_type)
```

**Expected output (approximate):**

```
spherical diffusivity [1/s] 3.333e-02
rectangular dims [m] (2.0, 1.5, 1.2)
rectangular distribution continuous_pdf
```

## Distribution types

- Default: `"discrete"`
- Also supported: `"continuous_pdf"`, `"particle_resolved"`
- Invalid values raise `ValueError` in `set_distribution_type` (and the strategy
  constructors).

## Validation expectations

- Negative or zero `wall_eddy_diffusivity`, `chamber_radius`, or any
  `chamber_dimensions` entry raises `ValueError`.
- `chamber_dimensions` must be exactly three positive values.
- Factory strategy names are case-insensitive but must be either "spherical" or
  "rectangular"; unknown names raise `ValueError`.

## Parameter reference

| Parameter | Applies to | Units | Notes |
| --- | --- | --- | --- |
| `wall_eddy_diffusivity` | Both builders | `1/s` | Positive; unit suffix supported |
| `chamber_radius` | Spherical | `m` | Positive; radius only |
| `chamber_dimensions` | Rectangular | `m` | Tuple `(x, y, z)` all positive |
| `distribution_type` | Both | `str` | Defaults to `"discrete"`; must be a supported value |

## See also

- Existing wall loss strategy walkthroughs: [Chamber Wall Loss examples](../Chamber_Wall_Loss/index.md)
- Dynamics overview: [Dynamics examples](index.md)
