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
transactions remain concrete-only in `nucleation.particle_source`. E6-F8 P1
provides an unexported, read-only Warp preflight and P2 privately plans
`E_pot=J*dt` demand after survival is included in `J`. It admits one shared
per-box demand against precursor inventory so planned removal is inventory-safe.
P2 commits only caller-owned planning, finalized-demand, and gate-diagnostic
sidecars. Private P3 then converts admitted demand times particle-box volume to
exact finite nonnegative `int32` provisional counts, reuses E6-F5 slot
diagnostics, and retains demand beyond free capacity while writing only the
count and fixed-shape diagnostic sidecars. It stages the selectable free-slot
prefix without activation or exhaustion policy and has no package export.
Conversion rejection occurs before sidecar writes; rollback is not promised
after a successful asynchronous diagnostic or P3 commit launch. Private P4
consumes immutable P2/P3 handoffs and caller-owned, same-device P4 and nested
E6-F6 sidecars. It selects resampling only when it can resolve a row's entire
deficit, then uses representative-volume scaling for remaining exhausted rows.
It derives finalized demand and counts from its mutable demand workspace and
current box volume, and emits finalized ascending free-slot prefixes with `-1`
tails. Expected all-box P4 rejections occur before workspace or primitive
writes, preserving particles, gas, P2/P3/P4 diagnostics, and nested scratch.
Once an E6-F6 primitive begins, its own planning/commit failure contract
applies; P4 intentionally provides no cross-primitive rollback. Activation,
source-mass or gas mutation, resizing, transfer, fallback, and E6-F9
integration remain deferred.

## Scientific Utilities

- Physical constants belong in `particula.util.constants`.
- Public numerical validation uses `particula.util.validate_inputs`.
- Chemical helper code lives under `particula/util/chemical/`.

## Documentation

Architecture guides and ADRs are under `.opencode/guides/architecture/`. Update
them when module boundaries, exported APIs, or major design patterns change.
