# Features Index

Development features tracked for particula. Features represent focused slices of
work, typically ~100 LOC per phase, that deliver user-facing functionality.

## Active Features

### Epic E3: Data Representation Refactor for Extensibility and GPU Backends

| ID | Name | Status | Priority | Phases |
|----|------|--------|----------|--------|
| E3-F1 | [Particle Data Container](E3-F1-particle-data-container.md) | Shipped | P1 | 4 |
| E3-F2 | [Gas Data Container](E3-F2-gas-data-container.md) | In Progress | P1 | 4 |
| E3-F3 | [Warp Integration and GPU Kernels](E3-F3-backend-warp-integration.md) | In Progress | P2 | 11 |
| E3-F4 | [Facade and Migration](E3-F4-facade-migration.md) | Planning | P1 | 5 |

### Standalone Features (Wall Loss)

| ID | Name | Status | Priority | Phases |
|----|------|--------|----------|--------|
| F1 | [Wall Loss Builders, Mixins, and Factory](wall-loss-builders-factory.md) | In Progress | P1 | 1 |
| F2 | [Rectangular Wall Loss Strategy](rectangular-wall-loss-strategy.md) | In Progress | P2 | 1 |
| F3 | [Charged/Electrostatic Wall Loss Strategy](charged-wall-loss-strategy.md) | In Progress | P1 | 2 |

### Epic E5: Non-Isothermal Condensation with Latent Heat

| ID | Name | Status | Priority | Phases |
|----|------|--------|----------|--------|
| E5-F1 | Latent Heat Strategy Pattern | Planning | P1 | 4 |
| E5-F2 | Non-Isothermal Mass Transfer Functions | Planning | P1 | 3 |
| E5-F3 | CondensationLatentHeat Strategy Class | Planning | P1 | 5 |
| E5-F4 | Builder, Factory, and Exports | Planning | P1 | 2 |
| E5-F5 | Validation and Integration Tests | Planning | P1 | 2 |
| E5-F6 | Documentation and Examples | Planning | P2 | 3 |
| E5-F7 | Warp/GPU Translation (Follow-on) | Planning | P2 | 3 |

### Standalone Features (End-to-End Simulation Notebooks)

| ID | Name | Status | Priority | Phases |
|----|------|--------|----------|--------|
| F5 | [Wildfire Plume Evolution Simulation](F5-wildfire-plume-simulation.md) | Planning | P2 | 3 |
| F6 | [Marine Aerosol Sea Spray Aging Simulation](F6-marine-aerosol-simulation.md) | Planning | P2 | 3 |

## Completed Features

### Epic E1: Staggered ODE Stepping for Particle-Resolved Condensation

| ID | Name | Completion Date |
|----|------|-----------------|
| E1-F1 | [Core Staggered Stepping Logic](completed/E1-F1-core-staggered-stepping.md) | 2026-01-07 |
| E1-F2 | [Batch-Wise Stepping Mode](completed/E1-F2-batch-stepping-mode.md) | 2026-01-07 |
| E1-F3 | [Builder and Factory Integration](completed/E1-F3-builder-factory-integration.md) | 2026-01-07 |
| E1-F4 | [Mass Conservation Validation](completed/E1-F4-mass-conservation-validation.md) | 2026-01-07 |
| E1-F5 | [Stability and Performance Benchmarks](completed/E1-F5-stability-performance-benchmarks.md) | 2026-01-07 |
| E1-F6 | [Documentation and Examples](completed/E1-F6-documentation-examples.md) | 2026-01-07 |

### Epic E2: Activity and Equilibria Strategy-Builder-Factory Refactor

| ID | Name | Completion Date |
|----|------|-----------------|
| E2-F1 | [Activity Strategy Refactor](completed/E2-F1-activity-strategy-refactor.md) | 2026-01-21 |
| E2-F2 | [Equilibria Runnable Refactor](completed/E2-F2-equilibria-runnable-refactor.md) | 2026-01-21 |
| E2-F3 | [Integration and Documentation](completed/E2-F3-integration-documentation.md) | 2026-01-21 |

### Standalone Features

| ID | Name | Completion Date |
|----|------|-----------------|
| F4 | [WallLoss Runnable Process](completed/wall-loss-runnable.md) | 2025-12-18 |
| F7 | [Cloud Chamber Injection Cycles Simulation](completed/F7-cloud-chamber-cycles-simulation.md) | 2026-01-20 |


## Next Available ID

**Standalone:** F8
