# Particula

A simple, fast, and powerful particle simulator for aerosol science.

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
| Contributing              | [contribute/](docs/contribute/CONTRIBUTING.md) |

### Examples by Topic

- **Aerosol** — Building and inspecting aerosol objects
- **Dynamics** — Coagulation, condensation, wall loss simulations
- **Equilibria** — Gas-particle partitioning calculations
- **Gas Phase** — Vapor pressure, species properties
- **Particle Phase** — Size distributions, optical properties
- **Simulations** — Full end-to-end scientific scenarios

### Featured Examples

- [**Aerosol Tutorial**](docs/Examples/Aerosol/Aerosol_Tutorial.ipynb) — Learn how to build gas species, atmospheres, particle distributions, and combine them into an `Aerosol` object.

- [**Organic Partitioning & Coagulation**](docs/Examples/Simulations/Notebooks/Organic_Partitioning_and_Coagulation.ipynb) — Full simulation of secondary organic aerosol (SOA) formation from 10 organic vapors, followed by Brownian coagulation over 10 minutes.

## Features

- **Gas & Particle Phases** — Full thermodynamic modeling with swappable strategies
- **Dynamics** — Coagulation, condensation, wall loss, dilution
- **Flexible Representations** — Discrete bins, continuous PDF, particle-resolved
- **Builder Pattern** — Clean, validated object construction with unit conversion
- **Composable Processes** — Chain runnables with `|` operator

## Citation

If you use Particula in your research, please cite:
> Particula [Computer software]. DOI: [10.5281/zenodo.6634653](https://doi.org/10.5281/zenodo.6634653)

## License

MIT
