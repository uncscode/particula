# Testing Strategy

Every implementation phase ships co-located tests in the same change. Coverage
thresholds are never lowered; changed code must retain at least 80% coverage.

## Per-Phase Coverage

- **P1:** CPU unit tests for defaults (`True`/`False`), all control combinations,
  precedence resolution, weighted inventory formulas, malformed values, and
  full-state no-mutation snapshots.
- **P2:** CPU deterministic tests for sparse/full boxes, tie breaks, exact slot
  clearing, number/species-mass/charge conservation, repeated plans, and
  predeclared radius/composition-moment bounds.
- **P3:** `particula/gpu/kernels/tests/exhaustion_test.py` compares Warp CPU with
  the independent NumPy oracle; validates sidecar identity/shape/dtype/device,
  failure ordering, and optional CUDA clean skips.
- **P4:** CPU/Warp tests for per-box scale factors, reciprocal weight updates,
  allowed bounds, unaffected boxes/fields, and represented-inventory parity.
- **P5:** Integration tests cover enough-capacity bypass, resampling-only,
  scaling-only, resampling-sufficient, resampling-then-scaling, both-off
  exhausted error, and unsatisfiable demand without any write or truncation.
- **P6:** Multi-box/multi-species matrix covers zero/sparse/full/over-capacity
  states, repeated calls, downstream-shaped requests, CPU/Warp tolerances, and
  source-plus-particle conservation.
- **P7:** Link, import-snippet, equation, shape-table, and supported/deferred
  boundary validation.

## Assertions and Tolerances

- Integer counts, policy codes, indices, sentinels, identities, shapes, dtypes,
  devices, and untouched state are exact.
- Number/species-mass/charge inventory uses independent float64 reductions with
  explicit `rtol`/`atol` recorded in tests; mixed scales receive species-level
  checks so large particles cannot hide small-particle loss.
- Distribution preservation uses named moments and thresholds fixed in P1, not
  exact slot-by-slot identity after resampling.
- Invalid calls snapshot particles, volume, requests, diagnostics, work buffers,
  and any RNG state before asserting equality.

Focused tests use the repository `*_test.py` convention. Full fast pytest, Ruff,
mypy, and documentation checks run before completion.
