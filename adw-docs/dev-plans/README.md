# Development Plans

Feature and maintenance development plans tracked for particula. Each entry
follows the appropriate template to capture motivation, scope, phases, testing,
and rollout.

## Indexes

- [Epics Index](epics/index.md) — All epics with status and next available ID
- [Features Index](features/index.md) — All features with status and next
  available ID
- [Maintenance Index](maintenance/index.md) — All maintenance plans with status

## Epics

- [E1: Staggered ODE Stepping for Particle-Resolved Condensation][epic-e1]
  — Status: Completed (P2)
  - Scope: Staggered ODE stepping framework for particle-resolved condensation
    with three modes (half-step, random, batch) for improved stability and
    mass conservation.
  - Features: [E1-F1][e1-f1], [E1-F2][e1-f2], [E1-F3][e1-f3], [E1-F4][e1-f4],
    [E1-F5][e1-f5], [E1-F6][e1-f6]

- [E3: Data Representation Refactor for Extensibility and GPU Backends][epic-e3]
  — Status: Planning
  - Scope: Isolate data representation from classes/methods using dataclass
    pattern. Enable extensibility (add fields without breaking simulations) and
    GPU acceleration via NVIDIA Warp for particle-resolved condensation and
    Brownian coagulation.
  - Features: [E3-F1][e3-f1], [E3-F2][e3-f2], [E3-F3][e3-f3], [E3-F4][e3-f4]

- [E4: Probabilistic Particle-Resolved Representation][epic-e4]
  — Status: Planning
  - Scope: New particle representation combining particle-resolved accuracy with
    super-droplet efficiency. Each computational particle represents a population
    with masses uniformly distributed in a configurable window (default 10%
    width). Enables 100-1000x fewer computational particles while preserving
    distribution information.
  - Features: [E4-F1][e4-f1], [E4-F2][e4-f2], [E4-F3][e4-f3], [E4-F4][e4-f4],
    [E4-F5][e4-f5], [E4-F6][e4-f6], [E4-F7][e4-f7], [E4-F8][e4-f8], [E4-F9][e4-f9]

- [E5: Non-Isothermal Condensation with Latent Heat][epic-e5]
  — Status: Planning
  - Scope: Non-isothermal condensation strategy accounting for latent heat of
    vaporization during mass transfer. Includes latent heat strategy pattern
    (constant, linear, power-law), thermal resistance correction on mass
    transfer rate, energy tracking to gas phase, and builder/factory
    integration. Python-native first, Warp/GPU follow-on.
  - Features: [E5-F1][e5-f1], [E5-F2][e5-f2], [E5-F3][e5-f3], [E5-F4][e5-f4],
    [E5-F5][e5-f5], [E5-F6][e5-f6], [E5-F7][e5-f7]

## Feature Plans

### Epic E1: Staggered Condensation Features

- [E1-F1: Core Staggered Stepping Logic][e1-f1] — Status: Completed (P2)
  - Scope: `CondensationIsothermalStaggered` class with theta modes and two-pass
    stepping algorithm.
- [E1-F2: Batch-Wise Stepping Mode][e1-f2] — Status: Completed (P2)
  - Scope: Gauss-Seidel batch stepping with configurable batch count.
- [E1-F3: Builder and Factory Integration][e1-f3] — Status: Completed (P2)
  - Scope: Builder and factory support for staggered condensation strategy.
- [E1-F4: Mass Conservation Validation][e1-f4] — Status: Completed (P2)
  - Scope: Comprehensive test suites validating mass conservation properties.
- [E1-F5: Stability and Performance Benchmarks][e1-f5] — Status: Completed
  (P3)
  - Scope: Benchmark tests for stability and performance characteristics (P2 performance benchmarks landed; issue #137).
- [E1-F6: Documentation and Examples][e1-f6] — Status: Completed (P3)
  - Scope: Docstrings, Jupyter notebook examples, and dev-docs updates.

### Epic E5: Non-Isothermal Condensation Features

- [E5-F1: Latent Heat Strategy Pattern][e5-f1] — Status: Planning
  - Scope: `LatentHeatStrategy` ABC with constant, linear, and power-law
    implementations plus builder/factory integration.
- [E5-F2: Non-Isothermal Mass Transfer Functions][e5-f2] — Status: Planning
  - Scope: Thermal resistance factor and non-isothermal mass transfer rate
    pure functions with energy tracking.
- [E5-F3: CondensationLatentHeat Strategy Class][e5-f3] — Status: Planning
  - Scope: New condensation strategy with latent heat correction and energy
    diagnostics.
- [E5-F4: Builder, Factory, and Exports][e5-f4] — Status: Planning
  - Scope: Builder, factory registration, and namespace exports for latent
    heat condensation.
- [E5-F5: Validation and Integration Tests][e5-f5] — Status: Planning
  - Scope: Mass conservation, isothermal parity, and physical validation
    tests.
- [E5-F6: Documentation and Examples][e5-f6] — Status: Planning
  - Scope: Docstrings, notebook examples, and dev-docs updates.
- [E5-F7: Warp/GPU Translation][e5-f7] — Status: Planning
  - Scope: GPU kernel translation of latent heat functions and condensation
    step.

### Standalone Features (Wall Loss)

- [Wall Loss Builders, Mixins, and Factory][plan-wall-loss-builders] — Status:
  In Progress (P1, #818)
  - Scope: Builder/factory parity for wall loss with validation and unit
    conversion.
- [Rectangular Wall Loss Strategy][plan-rect] — Status:
  In Progress (P2, #817)
  - Scope: Rectangular chamber wall loss strategy with mirrored tests and
    exports.
- [Charged/Electrostatic Wall Loss Strategy][plan-charged-wall-loss] — Status:
  In Progress (P1, #821)
  - Scope: Charged wall loss with image-charge, optional E-field drift,
    builder/factory integration, neutral reduction path, docs/examples.
- [WallLoss Runnable Process][plan-wall-loss-runnable] — Status:
  Completed (P1, #819)
  - Scope: Runnable wrapping wall loss strategies with sub-step splitting,
    non-negative clamp, exports, and runnable-level tests.

### Standalone Features (End-to-End Simulation Notebooks)

- [Wildfire Plume Evolution Simulation][plan-wildfire] — Status: Planning (P2)
  - Scope: Multi-stage temperature profile simulation of wildfire smoke from
    near-source emission to regional transport. Demonstrates combined
    coagulation (Brownian + turbulent shear + sedimentation) and dilution.
- [Marine Aerosol Sea Spray Aging Simulation][plan-marine] — Status: Planning
  (P2)
  - Scope: NaCl + organic film sea spray evolution in marine boundary layer
    through cloud deck formation. Showcases BAT model activity coefficients
    and phase separation diagnostics.
- [Cloud Chamber Injection Cycles Simulation][plan-cloud-chamber] — Status:
  Completed (P4)
  - Scope: 4-cycle cloud droplet activation/deactivation in rectangular chamber
    with particle-resolved speciated mass tracking. Demonstrates wall loss,
    dilution, injection, and κ-dependent activation for different seed types.
  - Phases: P1 (#896) + P2 (#897) + P3 + P4 completed.

## Maintenance Plans

### Active Maintenance

- [Jupyter Notebook API Migration][plan-notebook-migration] — Status: Planning
  (P2)
  - Scope: Migrate all 46 documentation Jupyter notebooks to current API
    patterns (builders, factories, `get_*` methods). Update descriptions and
    validate with `run_notebook` tool.

- [Jupytext Notebook Sync - Full Migration][plan-jupytext-full] — Status:
  In Progress (P2)
  - Scope: Complete migration of remaining ~35 notebooks to Jupytext paired sync
    format. Implement pre-commit hooks and CI validation.
  - Progress: 10/15 phases completed (all notebook conversions done; P14-P15 remain)

### Completed Maintenance

- [Add Charge Support to add_concentration][plan-charge-add-concentration] —
  Status: Shipped (P2)
  - Scope: Enable `add_concentration()` to accept optional charge parameter for
    ion injection in coagulation simulations.

- [Jupytext Notebook Sync Migration (Pilot)][plan-jupytext-pilot] — Status:
  Completed (P2)
  - Scope: Pilot migration of 4 notebooks (Activity, Gas_Phase) to Jupytext
    paired sync format (`.py:percent`). Validate workflow with ADW tools.

- [Particle-Resolved Coagulation Fixes][plan-coag-fixes] — Status:
  Shipped (P1)
  - Scope: Fix duplicate-index mass/charge loss in `collide_pairs()` and
    `get_particle_resolved_update_step()`. Add opt-in direct kernel evaluation
    to bypass charge-blind interpolation. Move diagnostic tests into repo.

## References

- Parent wall loss epic: [#72](https://github.com/uncscode/particula/issues/72)

<!-- Epic Links -->
[epic-e1]: epics/completed/E1-staggered-condensation-stepping.md
[epic-e3]: epics/E3-data-representation-refactor.md
[epic-e4]: epics/E4-probabilistic-particle-resolved.md
[epic-e5]: epics/E5-non-isothermal-condensation.md

<!-- E3 Feature Links -->
[e3-f1]: features/E3-F1-particle-data-container.md
[e3-f2]: features/E3-F2-gas-data-container.md
[e3-f3]: features/E3-F3-backend-warp-integration.md
[e3-f4]: features/E3-F4-facade-migration.md

<!-- E1 Feature Links -->
[e1-f1]: features/completed/E1-F1-core-staggered-stepping.md
[e1-f2]: features/completed/E1-F2-batch-stepping-mode.md
[e1-f3]: features/completed/E1-F3-builder-factory-integration.md
[e1-f4]: features/completed/E1-F4-mass-conservation-validation.md
[e1-f5]: features/completed/E1-F5-stability-performance-benchmarks.md
[e1-f6]: features/completed/E1-F6-documentation-examples.md

<!-- E4 Feature Links -->
[e4-f1]: features/E4-F1-core-strategy-data.md
[e4-f2]: features/E4-F2-distribution-shape-interface.md
[e4-f3]: features/E4-F3-extended-representation.md
[e4-f4]: features/E4-F4-probabilistic-condensation.md
[e4-f5]: features/E4-F5-probabilistic-coagulation.md
[e4-f6]: features/E4-F6-split-merge-maintenance.md
[e4-f7]: features/E4-F7-builder-factory.md
[e4-f8]: features/E4-F8-representation-conversion.md
[e4-f9]: features/E4-F9-documentation-examples.md

<!-- E5 Feature Links -->
[e5-f1]: features/E5-F1-latent-heat-strategy.md
[e5-f2]: features/E5-F2-non-isothermal-mass-transfer.md
[e5-f3]: features/E5-F3-condensation-latent-heat-strategy.md
[e5-f4]: features/E5-F4-builder-factory-exports.md
[e5-f5]: features/E5-F5-validation-integration-tests.md
[e5-f6]: features/E5-F6-documentation-examples.md
[e5-f7]: features/E5-F7-warp-gpu-translation.md

<!-- Standalone Feature Links -->
[plan-wall-loss-builders]: features/wall-loss-builders-factory.md
[plan-rect]: features/rectangular-wall-loss-strategy.md
[plan-charged-wall-loss]: features/charged-wall-loss-strategy.md
[plan-wall-loss-runnable]: features/wall-loss-runnable.md
[plan-wildfire]: features/F5-wildfire-plume-simulation.md
[plan-marine]: features/F6-marine-aerosol-simulation.md
[plan-cloud-chamber]: features/completed/F7-cloud-chamber-cycles-simulation.md

<!-- Maintenance Links -->
[plan-notebook-migration]: maintenance/M2-notebook-api-migration.md
[plan-charge-add-concentration]: maintenance/M1-add-concentration-charge-support.md
[plan-jupytext-pilot]: maintenance/M3-jupytext-notebook-sync.md
[plan-jupytext-full]: maintenance/M4-jupytext-full-migration.md
[plan-coag-fixes]: maintenance/M6-particle-resolved-coagulation-fixes.md
