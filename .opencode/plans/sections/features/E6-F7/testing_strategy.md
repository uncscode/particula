# Testing Strategy

Every phase ships implementation and co-located fast tests. Files use the
`*_test.py` convention, the configured coverage threshold remains at least 80%,
and scientific expectations come from hand calculations or an independent
NumPy oracle rather than production helpers.

## Per-Phase Approach

- **P1:** `nucleation_strategies_test.py` verifies `J=A*C`, `J=K*C^2`, SI
  conversion, coefficient dimensions, exact boundaries, no-op gates,
  NaN/Inf/negative rejection, and cited fixtures.
- **P2:** `particle_source_test.py` covers one/many boxes and species, each
  limiting species, exact inventory exhaustion, zero inputs, survival factors,
  represented packaging, diagnostics, and byte-for-byte input immutability.
- **P3:** Source interaction tests cover free/exact/sparse/full slots,
  resampling precedence, scaling fallback, both-off/unsatisfiable failures,
  deterministic order, no residual truncation, and all-box snapshots.
- **P4:** Builder/factory tests cover defaults, units, required domain and
  composition, dispatch, public imports, and unsupported model aliases.
- **P5:** `nucleation_runnable_test.py` verifies delegation,
  `time_step/sub_steps`, current-gas recomputation, exact no-ops, invalid
  substeps, returned identity, and composability.
- **P6:** `particula/integration_tests/nucleation_process_test.py` compares
  potential/admitted events and per-species total inventory with independent
  `float64` equations over repeated and substepped calls.
- **P7:** Validate links, citations, imports, equations, snippets, and execute
  the supported example/notebook where applicable.

## Required Invariants

- Per box/species, post particle represented mass plus post gas mass equals the
  pre total (target `rtol=1e-12`, `atol=1e-30`; any fixture-specific adjustment
  requires written numerical justification).
- Gas remains finite/nonnegative; admitted events never exceed potential events
  or participating species inventory.
- Zero time, coefficient, precursor, survival, and unsatisfied configured gate
  are exact no-ops with zero diagnostics.
- Rejected calls preserve particles, gas, diagnostics, work buffers, shapes,
  dtypes, and identities.
- Particle mass added equals gas mass removed independently for every species;
  aggregate-only checks are insufficient.

Focused suites are deterministic and fast. E6-F7 makes no stochastic CPU/GPU
sequence or performance claim.
