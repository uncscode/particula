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
- **Strategy-based API** (preferred):
  - `particula.dynamics.WallLossStrategy` – abstract base class for wall loss
    models.
  - `particula.dynamics.SphericalWallLossStrategy` – spherical chamber
    strategy.
  - `particula.dynamics.RectangularWallLossStrategy` – rectangular chamber
    strategy with `(x, y, z)` dimensions in meters.

Wall loss strategies operate directly on
`particula.particles.representation.ParticleRepresentation` instances and
support all three distribution types: "discrete", "continuous_pdf", and
"particle_resolved".

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

rate = rectangular_loss.rate(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
)

particle = rectangular_loss.step(
    particle=particle,
    temperature=298.15,
    pressure=101325.0,
    time_step=1.0,
)
```

See the online documentation for more examples and background theory.

