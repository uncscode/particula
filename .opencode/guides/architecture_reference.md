# Architecture Reference

**Project:** particula  
**Last Updated:** 2026-07-25

This reference summarizes the particula package structure and key architectural
conventions migrated from the legacy guide set.

## Package Map

```text
particula/
├── activity/          # Activity coefficients and phase separation
├── dynamics/          # Coagulation, condensation, wall loss
├── equilibria/        # Partitioning calculations
├── execution/         # Dependency-neutral execution selection
├── gas/               # Gas phase, species, vapor pressure
├── particles/         # Particle distributions and representations
├── util/              # Constants, validation, chemistry utilities
└── integration_tests/ # Cross-module integration tests
```

## Architectural Patterns

- Keep physics calculations in focused modules with clear units and citations.
- Use strategy, builder, and factory patterns where the package already uses
  them, especially in wall loss, vapor pressure, and representation code.
- Keep tests co-located with modules in `tests/` directories.
- Export public APIs deliberately through package `__init__.py` files.
- Keep validation close to public function boundaries.

## Execution Selection and Condensation State

`particula.execution` is a package. Its supported selection surface remains
the existing exact ten-name public export list; do not promote adapter modules
or state carriers through it or through top-level `particula`.

`particula.execution.adapters.condensation` contains concrete-only P2
condensation state carriers for future adapters. They retain caller-owned CPU
or resident Warp resources by identity and perform construction-time,
read-only metadata and ownership checks only. They do not select or execute an
adapter, transfer, allocate, or synchronize resources. Frozen carriers prevent
field rebinding, not mutation of retained caller-owned resources. Future
adapter authors remain responsible for resource lifetime, synchronization,
concurrency, and any post-launch mutation or rollback semantics.

## Wall Loss

Wall loss strategies live in `particula.dynamics.wall_loss` and are exported
through `particula.dynamics`.

Key concepts:

- `WallLossStrategy` is the abstract base class.
- `WallLoss` wraps a strategy, splits `time_step` across `sub_steps`, clamps
  concentrations nonnegative, and composes with other runnables via `|`.
- `ChargedWallLossStrategy` adds image-charge enhancement, optional electric
  field drift, and neutral fallback when particle charge and field are zero.
- Supported distribution types are `"discrete"`, `"continuous_pdf"`, and
  `"particle_resolved"`.

## Nucleation

The public CPU-only nucleation API is exported through `particula.dynamics`.
P4 provides immutable activation/kinetic potential-rate strategies,
source-selection metadata, builders, and a factory. P5 provides the one-box
`Nucleation` runnable and `NucleationCommitConfig`.

`Nucleation` preserves the legacy `Aerosol` and backing-data identities. It
uses equal, gas-coupled substeps and is atomic per attempted substep, not across
the complete call. P2 source-demand planning and P3 particle-source
transactions remain concrete-only in `nucleation.particle_source`.

The concrete `particula.gpu.kernels.nucleation` module implements the
package-exported direct `nucleation_step_gpu`: P1 preflight, P2 admission, P3
fixed-slot staging, P4 resampling-first/scaling-fallback, and fused P5
selected-slot/gas-transfer commit. Only the step is exported; configuration,
records, sidecars, and helpers remain concrete-only. Callers own conversion,
same-device fixed-shape sidecars, device placement, and synchronization; a
successful call returns the identical particle and gas containers. P3 retains
demand beyond free capacity and reports ascending free-slot prefixes with `-1`
tails. Public rejection before P4 primitive entry preserves particle and gas
state, while documented P2--P4 sidecars may already have been written; no
rollback is promised after E6-F6 primitive entry or P5 writer launch. The direct
step has no hidden transfer, CPU fallback, resize/compaction, GPU `Runnable`, or
E6-F9 integration.

## Scientific Utilities

- Physical constants belong in `particula.util.constants`.
- Public numerical validation uses `particula.util.validate_inputs`.
- Chemical helper code lives under `particula/util/chemical/`.

## Documentation

Architecture guides and ADRs are under `.opencode/guides/architecture/`. Update
them when module boundaries, exported APIs, or major design patterns change.
