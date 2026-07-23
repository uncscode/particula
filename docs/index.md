---
template: home.html
title: Particula
description: Predict Experiments, Expand Your Insights.
hide:
  - toc
---

# Particula

## What is Particula?

Particula is an open-source, Python-based aerosol simulator that bridges experimental data with computational models. It captures gas-particle interactions, transformations, and dynamics to power **predictive aerosol science**—so you can uncover deeper insights and accelerate progress.

---

## Why Use Particula?

Aerosols influence atmospheric science, air quality, and human health in powerful ways. Gaining insight into how they behave is essential for effective pollution control, accurate cloud formation modeling, and safer indoor environments. Particula provides a robust, flexible framework to simulate, analyze, and visualize aerosol processes with precision—empowering you to make breakthroughs and drive impactful science.

---

## How Does Particula Help You?

Whether you’re a researcher, educator, or industry expert, Particula is designed to **empower your aerosol work** by:

- **Harnessing ChatGPT integration** for real-time guidance, troubleshooting, and Q&A, [**here**](https://chatgpt.com/g/g-67b9dffbaa988191a4c7adfd4f96af65-particula-assistant).
- **Providing a Python-based API** for reproducible and modular simulations.
- **Building gas-phase properties** with builder/factory patterns (vapor
  pressure and latent heat) that support unit-aware setters and exports.
- **Supporting CPU latent-heat-corrected condensation diagnostics** with
  thermal resistance, latent-heat mass transfer rate utilities,
  latent-heat energy-density bookkeeping, and the
  `CondensationLatentHeat` strategy with latent-heat-corrected
  `mass_transfer_rate()`/`rate()` plus a `step()` that tracks per-step
  latent heat diagnostics. The bounded, low-level direct GPU condensation
   path optionally applies a latent-rate correction during each of its four
   equal substeps, with deterministic fp64 CPU-oracle/Warp parity coverage.
   Omitted latent heat, or a zero per-species value, retains that species' isothermal rate path. The direct hook couples each P2-finalized particle transfer to gas
   using particle concentration and has separate particle-mass, gas-
   concentration, and per-box/per-species inventory-conservation regressions.
    Broader temperature feedback and CPU-strategy/runnable-level support remain
    deferred; this bounded direct-kernel behavior does not establish those
    broader contracts.
- **Interrogating your experimental data** to validate and expand your impact.
- **Fostering open-source collaboration** to share ideas and build on each other’s work.

---

## Join the Community

We welcome contributions from scientists, developers, and students—and anyone curious about aerosol science! Whether you’re looking to ask questions, get help, or contribute fresh ideas, you’ve come to the right place.

Get more by posting on [GitHub Discussions](https://github.com/uncscode/particula/discussions) and tag any of the [contributors](https://github.com/uncscode/particula/graphs/contributors) using `@github-handle`.

- 💬 [**Ask questions** and **get help**](https://github.com/uncscode/particula/discussions/new?category=q-a).
- 🚀 [*Share your research*](https://github.com/uncscode/particula/discussions/new?category=show-and-tell) with the community to inspire others.
- 📣 [*Give us feedback.*](https://github.com/uncscode/particula/discussions/new?category=feedback)
- 🌟 **Contribute** to Particula by [*submitting pull requests*](https://github.com/uncscode/particula/pulls) or [*reporting issues*](https://github.com/uncscode/particula/issues) on GitHub.
- 🔗 Read our [**Contributing Guide**](contribute/CONTRIBUTING.md) to learn how you can make an impact.

We’re excited to collaborate with you! ✨

---

## Cite Particula in Your Research

Particula [Computer software]. [DOI: 10.5281/zenodo.6634653](https://doi.org/10.5281/zenodo.6634653)

---

## Get Started with Particula

[Setup Particula](Examples/Setup_Particula/index.md){ .md-button .md-button--primary }
[API Reference](https://uncscode.github.io/particula/API/){ .md-button }
[Examples](Examples/index.md){ .md-button }
[Theory](Theory/index.md){ .md-button }

---

### :simple-pypi: PyPI Installation
If your Python environment is already set up, install Particula directly from PyPI:
```shell
pip install particula
```

### :simple-condaforge: Conda Installation

Alternatively, you can install Particula using conda:
```shell
conda install -c conda-forge particula
```

If you are new to Python or plan on going through the Examples, head to [Setup Particula](Examples/Setup_Particula/index.md) for more comprehensive installation instructions.

### Quick Start Example

This “Quick Start Example” demonstrates a concise workflow for building an aerosol system in Particula and performing a single condensation step.

```python
import numpy as np
import particula as par

# 1. Build the GasSpecies for an organic vapor:
organic = (
    par.gas.GasSpeciesBuilder()
    .set_name("organic")
    .set_molar_mass(180e-3, "kg/mol")
    .set_vapor_pressure_strategy(
        par.gas.ConstantVaporPressureStrategy(1e2)  # Pa
    )
    .set_partitioning(True)
    .set_concentration(np.array([1e2]), "kg/m^3")
    .build()
)

# 2. Use AtmosphereBuilder to configure temperature, pressure, and species:
atmosphere = (
    par.gas.AtmosphereBuilder()
    .set_temperature(298.15, "K")
    .set_pressure(101325, "Pa")
    .set_more_partitioning_species(organic)
    .build()
)

# 3. Build the particle distribution:
#    Using PresetParticleRadiusBuilder, we set mode radius, GSD, etc.
particle = (
    par.particles.PresetParticleRadiusBuilder()
    .set_mode(np.array([100e-9]), "m")
    .set_geometric_standard_deviation(np.array([1.2]))
    .set_number_concentration(np.array([1e8]), "1/m^3")
    .set_density(1e3, "kg/m^3")
    .build()
)

# 4. Create the Aerosol combining the atmosphere and particle distribution:
aerosol = (
    par.AerosolBuilder()
    .set_atmosphere(atmosphere)
    .set_particles(particle)
    .build()
)

# 5. Define the isothermal condensation strategy:
condensation_strategy = par.dynamics.CondensationIsothermal(
    molar_mass=180e-3,  # kg/mol
    diffusion_coefficient=2e-5,  # m^2/s
    accommodation_coefficient=1.0,
)

# 6. Build the MassCondensation process:
process = par.dynamics.MassCondensation(condensation_strategy)

# 7. Execute the condensation process over 10 seconds:
result = process.execute(aerosol, time_step=10.0)

#   The result is an Aerosol instance with updated particle properties.
print(result)
```

---

### Feature deep-dives

- [Condensation strategy system](Features/condensation_strategy_system.md) —
  strategy-based condensation (simultaneous and staggered theta modes) with
  runnable pipelines.
- [Wall loss strategy system](Features/wall_loss_strategy_system.md) — chamber
  wall loss strategies with builders, factory, and runnable integration.
- [CPU dilution strategy system](Features/dilution_strategy_system.md) —
   construct supported processes with `par.dynamics.DilutionStrategy(coefficient)`
   and `par.dynamics.Dilution(strategy)`. The concrete helper boundary remains:
   `dilute_aerosol` and `get_dilution_step` are concrete-module-only helpers,
   not `particula.dynamics` public APIs. Run the
   [public API source example](https://github.com/Gorkowski/particula/blob/main/docs/Examples/cpu_dilution.py)
   with `python docs/Examples/cpu_dilution.py`.
- GPU dilution P1–P4 is a direct, low-level operation imported with
   `from particula.gpu.kernels import dilution_step_gpu`. It applies
   `c_new = c * exp(-alpha * time_step)` in place to particle and gas
   concentrations, where `alpha = Q / V` `[s^-1]`, returns the identical
   containers, and preserves all other caller-owned fields. Callers own
   CPU↔Warp transfers, device placement, and synchronization; there is no hidden
   transfer or CPU fallback. Coefficients are finite nonnegative scalars or
   same-device `wp.float64` Warp arrays shaped `(n_boxes,)`. Deterministic,
   read-only preflight may run validation scans that allocate or launch, but
   invalid calls have no update-kernel launch or caller mutation. Scalar-zero
   coefficients and zero time steps complete preflight and
   are write-free, no-update-kernel no-ops; validation scans may still allocate
   or launch. Warp CPU float64
   particle and gas comparisons use `rtol=1e-12, atol=0`; CUDA is optional and
   skips cleanly when unavailable. This is tolerance-based evidence, not
   bitwise parity. GPU runnables, orchestration, resizing, graph capture,
   autodiff, and performance claims remain deferred. See
    [Data containers and GPU foundations](Features/data-containers-and-gpu-foundations.md)
    for the complete contract.
- GPU neutral wall loss P4 is a direct fixed-shape boundary imported with
    `from particula.gpu.kernels import wall_loss_step_gpu`. It accepts only
    particle-resolved neutral configurations. Create
    `NeutralWallLossConfig` from `particula.gpu.kernels.wall_loss`; the
    configuration is intentionally not exported from `particula.gpu.kernels` or
    `particula.gpu`. It validates spherical or rectangular SI geometry, fixed
    `WarpParticleData` schema and domains, environment inputs, and optional RNG
    metadata. Successful nonzero calls evaluate bounded neutral coefficients
    and stochastically remove eligible fixed slots in place; zero time is a
    post-preflight write-free no-op. P4 derives one local draw from the seed and
    slot and does not initialize, advance, or otherwise mutate `rng_states`;
    P5 owns lifecycle semantics. Callers retain ownership of Warp transfers,
    device placement, synchronization, particle data, and any RNG sidecar.
    Preflight may run device validation scans and synchronize to read back scalar
    status, but it does not transfer or replace caller-owned buffers. Charged
    wall loss, a runnable API, hidden transfers or fallback, CPU/Warp stochastic
    parity, and P5/P6 behavior remain deferred.
- [Data containers and GPU foundations](Features/data-containers-and-gpu-foundations.md)
  — canonical reference for `ParticleData`, `GasData`, `EnvironmentData`,
   explicit CPU↔GPU transfer helpers, leading-axis shape conventions, the
   current shipped CPU/GPU support boundary, and caller-owned GPU sidecar state
   such as coagulation `rng_states` and condensation thermodynamics. Direct,
   particle-resolved GPU coagulation executes singleton masks `1`, `2`, `4`,
   and `8`; unordered two-way masks `3`, `5`, `6`, `9`, `10`, and `12`; and
   four-way mask `15` across Brownian, charged hard-sphere, SP2016
   sedimentation, and ST1956 turbulent shear. Non-turbulent three-way mask `7`
   rejects at capability preflight before particle metadata or enabled-term
   validation. Turbulent three-way masks `11`, `13`, and `14` proceed through
   particle metadata and enabled-term validation, then reject before downstream
   normalization, allocation, RNG setup, kernel launch, or mutation. Approved
   turbulent-shear masks require keyword-only positive, finite
   `turbulent_dissipation` (m²/s³) and `fluid_density` (kg/m³) Python or NumPy
   floating scalars, or active-device `wp.float64` `(n_boxes,)` Warp arrays.
   This is a direct-kernel-only contract: high-level `Runnable` support, CPU
   fallback, hidden state transfer, graph capture, performance claims, and
   broad accuracy claims remain out of scope. GPU condensation requires
   keyword-only
   `thermodynamics=ThermodynamicsConfig`,
  validates an active-device binary per-box/species `gas.partitioning` mask
  before mutation, and runs exactly four equal substeps. Disabled species and
  zero-concentration particle slots receive no transfer. Each finalized
  transfer updates particle mass and applies the matching weighted delta to
  `gas.concentration`, so later substeps read coupled gas inventory. It
  optionally accepts `latent_heat` rate correction and its write-only signed
  `energy_transfer` diagnostic, and refreshes caller-owned
   `WarpGasData.vapor_pressure` from the current device-resident temperature
   before mass transfer. For the supported low-level, direct-kernel-only
    walkthrough, use `python docs/Examples/gpu_direct_kernels_quick_start.py`.
    This path does not add high-level `Aerosol`/`Runnable` support, implicit
    simulation-state transfers or synchronization, automatic fallback or
    migration, or CPU-strategy/runnable parity. See the [GPU condensation command
     matrix](Features/data-containers-and-gpu-foundations.md#focused-reproduction-commands)
     for focused troubleshooting and reproduction commands. Entry-point
     validation can still perform synchronous device-to-host readbacks.
   For a focused supported coagulation route, the
    [direct GPU coagulation example](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_coagulation_direct.py)
   explicitly transfers `ParticleData`, runs two Brownian particle-resolved
   calls with caller-owned collision and persistent RNG sidecars on Warp CPU
   by default, then restores a CPU checkpoint. When Warp is unavailable or
   disabled, it runs no conversion or kernel and provides no CPU fallback.
   It makes no `Runnable`, CUDA, or performance claim.
- [ParticleData and GasData migration guide](Features/particle-data-migration.md)
  — migration workflow and before/after examples for moving from legacy facades
  to the canonical data-container contract documented in the foundation guide.
- [Data-oriented design and GPU roadmap](Features/Roadmap/data-oriented-gpu.md)
  — current schema inventory,
  [authoritative CPU/GPU field ownership policy](Features/Roadmap/data-oriented-gpu.md#authoritative-field-ownership-decisions),
  shipped coagulation RNG ownership and graph-capture setup guidance,
  and the [final downstream handoff map for sibling E2
  features](Features/Roadmap/data-oriented-gpu.md#final-downstream-handoff-map-for-sibling-features).
- [Mass Precision Recommendation Report](Features/Roadmap/mass-precision-study.md)
  — final E2-F6 policy report covering deterministic NPF-to-droplet GPU
  evidence, the accepted unchanged `fp64`/`wp.float64` production baseline,
  study-only candidate fidelity checks, executable P3 thresholds, clamp
  accounting, memory-footprint examples, and focused reproduction commands.
