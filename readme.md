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

## AI Assistants

[Particula Helper](docs/guide/ai-assistants/index.md) provides setup, simulation,
analysis, and debugging workflows for Codex/ChatGPT, Claude Code/Cowork, and
OpenCode. See the guide for native install commands and environment requirements.

## Quick Start

The following native example requires the selected E9 checkout (install it with
`pip install -e .`). It does not imply that an older published wheel contains
these migration APIs.

```python
import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import CondensationSpeciesMap

particles = par.particles.create_resolved_particle_data(
    masses=[1e-18, 2e-18], density=1000.0,
    concentration=[1.0, 1.0], volume=1e-6,  # counts and m³
)
gas = par.gas.GasData(
    name=["organic"], molar_mass=np.array([0.180]),
    concentration=np.array([[1e-6]]),  # kg/m³
    partitioning=np.array([True]),
)
environment = par.gas.EnvironmentData(
    temperature=[298.15], pressure=[101325.0], saturation_ratio=[[1.0]],
)
aerosol = par.Aerosol(particles=particles, gas=gas, environment=environment)

strategy = par.dynamics.CondensationIsothermal(
    molar_mass=gas.molar_mass.copy(),
    diffusion_coefficient=2e-5, accommodation_coefficient=1.0,
    activity_strategy=par.particles.ActivityIdealMass(),
    surface_strategy=par.particles.SurfaceStrategyVolume(density=1000.0),
    vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(1e-3),
    species_map=CondensationSpeciesMap(("organic",), ((0, 0),)),
)
condensation = par.dynamics.MassCondensation(strategy)
coagulation = par.dynamics.Coagulation(
    par.dynamics.BrownianCoagulationStrategy("particle_resolved")
)
aerosol = (condensation | coagulation).execute(
    aerosol, time_step=10.0, sub_steps=10
)
print(aerosol.particles.number_density)  # m⁻³, not raw counts
```

This single-box illustration retains explicit population and species meaning.
See [native access and replacement](docs/guide/data-model/ownership.md)
for ownership, validation and non-unit-volume inventory checks.


## Migration / What's New

See the [migration guide](./docs/Features/particle-data-migration/) for
current APIs, compatibility notes, and direct-kernel troubleshooting.

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

- [**Aerosol Tutorial**](./docs/examples/fundamentals/native_aerosol.ipynb) — Build native particle, full gas and environment containers and combine them into an `Aerosol` object.

- [**CPU Latent-Heat Condensation Bookkeeping**](./docs/examples/processes/condensation/condensation_latent_heat.ipynb)
  — CPU-only walkthrough showing diagnostic latent-heat bookkeeping from real
  condensation mass transfer with no temperature feedback.

- [**Direct-Warp GPU Nucleation**](./docs/examples/gpu/direct/gpu_direct_nucleation.py)
  — Explicit-transfer, fixed-capacity low-level example using the package-
  exported `nucleation_step_gpu`. It requires Warp and is separate from the
  CPU-only `Nucleation` runnable; it provides no CPU fallback.

- [**Organic Partitioning & Coagulation**](./docs/examples/simulations/organic_partitioning_and_coagulation.ipynb) — Illustrative organic partitioning followed by Brownian coagulation, including the retained binned method’s composition limitation; not a quantitative species forecast.

- [**Cloud Chamber Cycles**](./docs/examples/simulations/cloud_chamber_multi_cycle.ipynb)
  — Multi-cycle cloud droplet activation demonstrating κ-Köhler theory across
  3 seed compositions (Ammonium Sulfate, Sucrose, Mixed), comparing illustrative humidity responses. Dry-phase timestep oscillations
  remain unqualified for quantitative activation predictions.

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
