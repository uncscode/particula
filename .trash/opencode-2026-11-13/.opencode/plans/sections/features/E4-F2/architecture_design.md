# Architecture Design

## High-Level Design

```text
E4-F1 temperature + numeric vapor-pressure config
  -> refreshed gas.vapor_pressure[n_boxes, n_species]
particle masses + density + molar mass + numeric E4-F2 config
  -> on-device composition denominators
  -> ideal/kappa activity and effective surface tension
  -> Kelvin radius/term
  -> activity * pure vapor pressure * Kelvin term
  -> pressure delta -> existing mass-transfer path
```

Formula helpers belong in focused GPU property modules; orchestration,
preflight validation, and launch order remain in
`particula/gpu/kernels/condensation.py`. Species position is identity. Model
modes are int32 and physical parameters/results are fp64 fixed-shape arrays.

## Data / API / Workflow Changes
- **Data model:** No CPU or Warp container schema changes. A frozen,
  caller-owned `CondensationActivitySurfaceConfig` operation sidecar carries
  int32-compatible activity/surface selectors, water index, `kappas`, and an
  ordered molar-mass reference; E4-F1 owns the refreshed
  vapor-pressure buffer. Supported surface-tension modes are static
  per-species values and one global, single-phase volume-weighted value;
  phase-aware weighting is deferred.
- **API surface:** Positional per-species `surface_tension` remains the legacy
  source. `activity_surface=` is keyword-only and validated as a transaction
  before environment normalization, defaults, vapor-pressure refresh, output
  allocation, or launch. A configured static call uses the condensing-species
  tension; weighted mode precomputes one scalar tension per active particle.
  Unsupported selectors and invalid sidecar, environment, optional, or output
  inputs fail without caller-state mutation.
- **Workflow hooks:** The primitive remains directly importable through
  `particula.gpu.kernels`, callable repeatedly by E4-F3, and feeds E4-F4.

## Security & Compliance

There is no permission or network change. Robustness requires bounded Kelvin
exponents, finite/nonnegative state checks, no Python strings or strategy
objects in Warp containers, no hidden host transfer, and deterministic failure
before state mutation.
