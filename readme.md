# Overview

Particula is a Python-based aerosol particle simulator. Its goal is to provide a
robust aerosol simulation (including both gas and particle phases) that can be
used to answer scientific questions arising from experiments and research
endeavors.

The Particula website
[https://uncscode.github.io/particula](https://uncscode.github.io/particula)
contains the API reference, how-to guides, and tutorials.

## PyPI Installation

If your Python environment is already set up, you can install
[`particula` via pip](https://pypi.org/project/particula/) using the following
command:

``` bash
pip install particula
```

Or install via conda:

``` bash
conda install -c conda-forge particula
```

## Dynamics and wall loss

The `particula.dynamics` namespace collects time-dependent processes such as
dilution, condensation, coagulation, and wall loss.

For wall loss there are two complementary APIs:

- **Function-based rates** (legacy):
  - `particula.dynamics.get_spherical_wall_loss_rate(...)`
  - `particula.dynamics.get_rectangle_wall_loss_rate(...)`
  - `particula.dynamics.get_charged_wall_loss_rate(...)` – charged/image-charge
    wall loss helper with optional electric-field drift.
- **Strategy-based API** (preferred):
  - `particula.dynamics.WallLossStrategy` – abstract base class for wall loss
    models.
  - `particula.dynamics.SphericalWallLossStrategy` – spherical chamber
    strategy.
  - `particula.dynamics.RectangularWallLossStrategy` – rectangular chamber
    strategy with `(x, y, z)` dimensions in meters.
  - `particula.dynamics.ChargedWallLossStrategy` – charged wall loss for
    spherical or rectangular chambers; includes image-charge enhancement even
    when `wall_potential=0`, optional `wall_electric_field` drift, and reduces
    to the neutral coefficient when charge and field are zero.

Wall loss strategies operate directly on
`particula.particles.representation.ParticleRepresentation` instances and
support all three distribution types: "discrete", "continuous_pdf", and
"particle_resolved". The charged strategy uses particle charge; when charge and
`wall_electric_field` are zero, it reduces to the neutral coefficient while
preserving image-charge effects when `wall_potential` is zero.

```python
import particula as par

# Assume `particle` is a ParticleRepresentation
spherical_loss = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=0.001,  # m^2/s
    chamber_radius=0.5,  # m
    distribution_type="discrete",
)

rectangular_loss = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=0.001,  # m^2/s
    chamber_dimensions=(1.0, 0.5, 0.3),  # m
    distribution_type="discrete",
)

charged_loss = par.dynamics.ChargedWallLossStrategy(
    wall_eddy_diffusivity=0.001,  # m^2/s
    chamber_geometry="rectangular",
    chamber_dimensions=(1.0, 0.5, 0.3),  # m
    wall_potential=0.05,  # V (image-charge active even if set to 0)
    wall_electric_field=(50.0, 0.0, 0.0),  # V/m; set to 0 to disable drift
    distribution_type="discrete",
)

rate = charged_loss.rate(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
)

particle = charged_loss.step(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)
```

Wall loss builders mirror other dynamics builders and handle unit conversion and
validation for geometry, diffusivity, and charged parameters (`wall_potential`,
`wall_electric_field`). The charged builder requires `chamber_geometry` plus a
matching size field (`chamber_radius` for spherical or `chamber_dimensions` for
rectangular). The factory accepts `strategy_type="charged"` with the same
parameters if you prefer name-based selection.

```python
import particula as par

builder = (
    par.dynamics.ChargedWallLossBuilder()
    .set_wall_eddy_diffusivity(0.001, "m^2/s")
    .set_chamber_geometry("rectangular")
    .set_chamber_dimensions((1.0, 0.5, 0.3), "m")
    .set_wall_potential(0.05, "V")
    .set_wall_electric_field((50.0, 0.0, 0.0), "V/m")
    .set_distribution_type("particle_resolved")
)
charged_loss = builder.build()

factory = par.dynamics.WallLossFactory()
charged_from_factory = factory.get_strategy(
    strategy_type="charged",
    parameters={
        "chamber_geometry": "spherical",
        "chamber_radius": 0.5,
        "wall_eddy_diffusivity": 0.001,
        "wall_potential": 0.0,
        "wall_electric_field": 0.0,
        "distribution_type": "continuous_pdf",
    },
)
```

When charges and `wall_electric_field` are zero, charged wall loss matches the
neutral coefficient; non-zero charge retains image-charge effects even when
`wall_potential` is zero.

See the online documentation for more examples and background theory. For a complete
walkthrough comparing spherical and rectangular chambers (with coagulation/condensation
chaining), see the
[Wall Loss Tutorial notebook](docs/Examples/Chamber_Wall_Loss/Notebooks/Wall_Loss_Tutorial.ipynb).

