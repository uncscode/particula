# Particula

A simple, fast, and powerful particle simulator for aerosol science.

**Requires:** Python 3.12+

[Documentation](https://uncscode.github.io/particula) | [Examples](https://uncscode.github.io/particula/Examples/) | [PyPI](https://pypi.org/project/particula/)

## Installation

```bash
pip install particula
```

or via conda:

```bash
conda install -c conda-forge particula
```

## Quick Start

```python
import particula as par

# Build an aerosol system
aerosol = (
    par.AerosolBuilder()
    .set_atmosphere(atmosphere)
    .set_particles(particles)
    .build()
)

# Run dynamics (chainable with | operator)
process = par.dynamics.Condensation(strategy) | par.dynamics.Coagulation(strategy)
aerosol = process.execute(aerosol, time_step=10, sub_steps=1000)
```

## Migration / What's New

For migration details and updated API mappings, see the canonical guide:
[ParticleData and GasData Migration Guide](./docs/Features/particle-data-migration.md).
Legacy facades remain available, with deprecation planned for v0.3.0.
`EnvironmentData` now also participates in the public Warp CPU↔GPU helpers via
`particula.gpu.{to_warp_environment_data, from_warp_environment_data}` for
single-box and multi-box round trips.
GPU kernel entry points `condensation_step_gpu` and `coagulation_step_gpu`
now accept scalar `temperature` / `pressure` inputs, per-box Warp arrays with
shape `(n_boxes,)`, hybrid scalar-plus-Warp-array direct inputs when
`environment` is omitted, or a `WarpEnvironmentData` via the keyword-only
`environment=` parameter.
Mixed scalar-plus-environment calls still fail early by design. Explicit
environment inputs must match the particle/gas device and use `(n_boxes,)`
temperature and pressure arrays. All accepted temperature, pressure, and
coagulation volume inputs are validated as positive finite physical values
before launch.

## Code Structure

```
particula/
├── gas/           # Gas phase: species, vapor pressure, atmosphere
├── particles/     # Particle representations & distributions
├── dynamics/      # Time-dependent processes
│   ├── coagulation/
│   ├── condensation/
│   └── wall_loss/
├── activity/      # Activity coefficients, phase separation
├── equilibria/    # Gas-particle partitioning
└── util/          # Constants, validation, unit conversion
```

## Documentation Guide

| Looking for...            | Go to                                      |
|---------------------------|--------------------------------------------|
| Tutorials & walkthroughs  | [Examples/](https://uncscode.github.io/particula/Examples/) |
| Scientific background     | [Theory/](https://uncscode.github.io/particula/Theory/) |
| API reference             | [Full Docs](https://uncscode.github.io/particula) |
| Contributing              | [contribute/](./docs/contribute/CONTRIBUTING.md) |

### Examples by Topic

- **Aerosol** — Building and inspecting aerosol objects
- **Dynamics** — Coagulation, condensation, wall loss simulations
- **Equilibria** — Gas-particle partitioning calculations
- **Gas Phase** — Vapor pressure, species properties
- **Particle Phase** — Size distributions, optical properties
- **Simulations** — Full end-to-end scientific scenarios

### Featured Examples

- [**Aerosol Tutorial**](./docs/Examples/Aerosol/Aerosol_Tutorial.ipynb) — Learn how to build gas species, atmospheres, particle distributions, and combine them into an `Aerosol` object.

- [**Organic Partitioning & Coagulation**](./docs/Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.ipynb) — Full simulation of secondary organic aerosol (SOA) formation from 10 organic vapors, followed by Brownian coagulation over 10 minutes.

- [**Cloud Chamber Cycles**](./docs/Examples/Simulations/Notebooks/Cloud_Chamber_Multi_Cycle.ipynb)
  — Multi-cycle cloud droplet activation demonstrating κ-Köhler theory across
  3 seed compositions (Ammonium Sulfate, Sucrose, Mixed), showing how
  hygroscopicity affects activation at different supersaturations.

## Features

- **Gas & Particle Phases** — Full thermodynamic modeling with swappable strategies
- **Dynamics** — Coagulation, condensation, wall loss, dilution
- **Flexible Representations** — Discrete bins, continuous PDF, particle-resolved
- **Builder Pattern** — Clean, validated object construction with unit conversion
- **Composable Processes** — Chain runnables with `|` operator
- **Condensation Utilities** — Non-isothermal helpers via
  `particula.dynamics.get_thermal_resistance_factor`,
  `particula.dynamics.get_mass_transfer_rate_latent_heat`, and
  `particula.dynamics.get_latent_heat_energy_released`
- **Condensation Strategies** — `CondensationIsothermal` plus
  `CondensationLatentHeat` with latent-heat-corrected
  `mass_transfer_rate()`/`rate()` and `step()` energy tracking via
  `last_latent_heat_energy`, with optional `dynamic_viscosity` override
- **Latent Heat Factories** — Build constant, linear, and power-law latent heat
  strategies via `particula.gas.LatentHeatFactory` with unit-aware builders and
  gas-phase exports for upcoming non-isothermal workflows

## Citation

If you use Particula in your research, please cite:
> Particula [Computer software]. DOI: [10.5281/zenodo.6634653](https://doi.org/10.5281/zenodo.6634653)

## License

MIT
