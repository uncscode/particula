# Condensation Stiffness Study Baseline

This note defines the reusable baseline for future GPU condensation stiffness
work. It is intentionally foundational: it names deterministic study cases,
shared metric language, and explicit pass/fail rules without publishing measured
timestep bounds or integrator recommendations.

## Current Runtime Scope

- Production path: explicit fixed-step GPU condensation.
- Particle update: particle masses are clamped to remain non-negative.
- Gas update: the current Warp path is particle-only and does not yet update gas
  concentrations during production condensation.
- Baseline backend for this phase: `np.float64` inputs and Warp CPU execution.
- Accepted environment inputs: scalar `temperature` and `pressure`, or direct
  Warp arrays with shape `(n_boxes,)`.
- Study cases are deterministic fixed-shape builders that should be recreated
  from clean inputs after failed validation or intentionally unstable runs.

## Shared Case Catalog

The executable test baseline defines three named regimes:

- `nanometer`
- `accumulation_mode`
- `droplet_like`

Each `CondensationStiffnessCase` declares:

- `name`
- `n_boxes`
- `n_particles`
- `n_species`
- `time_step`
- scalar baseline `temperature` and `pressure`
- deterministic scaling for particle mass, gas concentration, and vapor
  pressure
- optional zero-mass or zero-concentration particle edges

The important contract is fixed shape, not random sampling. Every case rebuilds
deterministic particle, gas, vapor-pressure, and environment inputs with
explicit `(n_boxes, n_particles, n_species)` and `(n_boxes,)` dimensions.

## Shared Metrics and Classification

The baseline tests use the following helper concepts:

- `_particle_mass_is_nonnegative`: all post-step particle masses are `>= 0`.
- `_particle_values_are_finite`: particle/gas/support arrays remain finite.
- `_fractional_mass_change_per_bin`: fractional change is computed only for
  positive initial masses, so zero-initial-mass bins need a separate stability
  check.
- `_zero_mass_entries_remain_stable`: zero-initial-mass bins remain at zero so
  the zero-mass edge stays deterministic; helper inputs must have matching
  shapes.
- `_validate_stiffness_case_metadata`: declared case metadata must match array
  shape and `np.float64` dtype expectations.
- `_classify_particle_only_condensation_stiffness`: returns a stable/unstable
  result for the current particle-only GPU path and marks the
  `particle_only_update` caveat explicitly.

### Boundary Semantics

- Fractional-mass-change thresholds are inclusive: exact equality remains
  `stable`.
- Zero-initial-mass growth remains `unstable` even though fractional change is
  reported only for positive initial masses.
- Declared shape mismatches fail validation.
- Declared dtype mismatches fail validation.
- Zero-mass entries are treated as stable only when they remain unchanged.
- Accepted explicit `environment=...` inputs with `(n_boxes,)` Warp arrays match
  accepted direct `(n_boxes,)` Warp-array inputs and should not mutate
  caller-owned temperature or pressure arrays.

## What This Phase Does Not Publish

This baseline does **not** publish:

- measured stable timestep limits
- broad timestep sweep tables
- candidate integrator comparisons
- gas-coupled conservation claims that the current production path does not yet
  satisfy

Later phases can measure explicit Warp behavior against this shared baseline
without redefining case shapes, metric names, or threshold meaning.
