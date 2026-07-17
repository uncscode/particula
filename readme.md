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

For the [Canonical low-level direct-condensation contract](./docs/Features/data-containers-and-gpu-foundations.md),
see the container, shape, and CPU↔GPU transfer boundaries.
For a runnable low-level walkthrough, run `python
docs/Examples/gpu_direct_kernels_quick_start.py` to see explicit
`to_warp_*` / `from_warp_*` boundaries, lazy kernel imports from
`particula.gpu.kernels`, two direct condensation calls with reused scratch
buffers, latent-heat, and energy sidecars on Warp's CPU backend by default.
For migration details and updated API mappings, see the
[ParticleData and GasData Migration Guide](./docs/Features/particle-data-migration.md).
Legacy facades remain available, with deprecation planned for v0.3.0.
`EnvironmentData` now also participates in the public Warp CPU↔GPU helpers via
`particula.gpu.{to_warp_environment_data, from_warp_environment_data}` for
single-box and multi-box round trips.
Import GPU kernel entry points `condensation_step_gpu` and
`coagulation_step_gpu` from `particula.gpu.kernels`. Top-level
`particula.gpu` remains the transfer/context-helper surface and does not
re-export those direct kernel step functions. The kernel entry points accept
scalar `temperature` / `pressure` inputs, per-box Warp arrays with shape
`(n_boxes,)`, hybrid scalar-plus-Warp-array direct inputs when `environment`
is omitted, or a `WarpEnvironmentData` via the keyword-only `environment=`
parameter.
Mixed scalar-plus-environment calls still fail early by design. Explicit
environment inputs must match the particle/gas device and use `(n_boxes,)`
temperature and pressure arrays. All accepted temperature, pressure, and
coagulation volume inputs are validated as positive finite physical values
before launch.

`coagulation_step_gpu` accepts an optional keyword-only `mechanism_config`
from `particula.gpu.kernels.coagulation`; this configuration API is not
re-exported from `particula.gpu.kernels`. Omitting it preserves the Brownian,
particle-resolved path. The low-level entry point also supports exact
charged-hard-sphere-only and canonical Brownian-plus-charged
`particle_resolved` configurations. The combined configuration accepts either
requested mechanism order and uses one shared stochastic selection path.
Malformed configurations, unsupported distributions, and deferred mechanisms
fail during host-side preflight before runtime state is accessed, allocated,
mutated, reseeded, or launched.

`condensation_step_gpu` additionally requires a keyword-only
`ThermodynamicsConfig` through `thermodynamics=`. After all inputs and optional
buffers validate, each successful call refreshes the caller-owned,
device-resident `WarpGasData.vapor_pressure` from the current normalized
per-box temperature before condensation mass transfer. This overwrite makes a
previous vapor-pressure buffer stale by design. Scalar temperatures, direct
Warp temperature arrays, and `WarpEnvironmentData` are supported; non-float64
temperature arrays are cast on-device for the refresh without a host
vapor-pressure transfer.

Caller-owned thermodynamics configurations and optional mass-transfer buffers
may be reused across calls. A failed preflight, including missing or
device-incompatible thermodynamics, leaves caller-owned simulation and output
buffers unchanged.

Before a direct condensation call can mutate state, `gas.partitioning` must be
an active-device, binary `wp.int32` mask shaped `(n_boxes, n_species)`.
Disabled species and zero-concentration particle slots receive no transfer.
Invalid masks, and invalid optional P2-only reduction/scale/accumulator scratch
sidecars, raise `ValueError` without changing caller-owned state. The P2-only
sidecars finalize inventory-limited transfers during each substep. The direct
kernel updates particle masses and applies the matching particle-concentration-
weighted delta to `gas.concentration` after each finalized transfer.

Each successful direct condensation call executes exactly four equal
`time_step / 4.0` substeps. A supplied scratch work buffer retains the final
raw P1 proposal, while the returned total (and optional energy diagnostic) use
the P2-finalized transfers accumulated over the whole call. Vapor-pressure
refresh depends on temperature, not gas concentration; subsequent
mass-transfer proposals read the gas concentration coupled by prior substeps.

The direct condensation kernel also accepts optional keyword-only,
active-device `wp.float64` sidecars: `latent_heat`, shaped `(n_species,)`, and
the caller-owned write-only `energy_transfer`, shaped `(n_boxes, n_species)`.
Nonzero latent heat applies the per-species rate correction during each of the
four fixed substeps; omitting it, or setting a species entry to zero, preserves
that species' isothermal rate path. `energy_transfer` requires `latent_heat`
and is overwritten after successful preflight with signed, whole-call
P2-finalized transfer times latent heat. It remains caller-owned rather than
becoming a third return item; the canonical contract defines its ownership,
units, and direct-step limits.
For focused troubleshooting and reproduction commands, see the [GPU condensation command matrix](./docs/Features/data-containers-and-gpu-foundations.md#focused-reproduction-commands).

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

- [**CPU Latent-Heat Condensation Bookkeeping**](./docs/Examples/Dynamics/Condensation/Condensation_Latent_Heat.ipynb)
  — CPU-only walkthrough showing diagnostic latent-heat bookkeeping from real
  condensation mass transfer with no temperature feedback.

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
