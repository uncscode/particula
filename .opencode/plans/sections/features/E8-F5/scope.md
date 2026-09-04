# Scope

E8-F5 adds full-loop validation fixtures and tests for identical multi-timestep
CPU, uncaptured Warp, and captured CUDA configurations. The deliverable verifies
observable state, accounting, stochastic lifecycle, and rejection semantics; it
does not introduce a new scheduler or change scientific models.

Issue #1575 delivered the P1 CPU-only foundation in
`particula/execution/tests/captured_full_loop_test.py`: a deterministic immutable
two-box fixture, independent NumPy oracle, diagnostic and inventory assertions,
no-work coverage, and validation rejection coverage. Uncaptured Warp and
captured CUDA parity remain later phases.

## In Scope

- Independent NumPy/CPU expected-state and inventory calculations.
- Identical fixed-shape, multi-box process configurations across all three paths.
- Per-field deterministic tolerances and separate tight conservation assertions.
- GAS and PARTICLES closed-map communication, optional volume evolution, and all
  registered resident diagnostic outputs.
- Coagulation and wall-loss RNG advancement, nonaliasing, explicit reset, and
  supported checkpoint/restart continuation evidence.
- Shape, device, schedule, communication-map, resource-identity, lifecycle, and
  stale-handle rejection before captured launch.
- Warp CPU as the required installed-Warp uncaptured baseline and CUDA capture
  as pass-or-clean-skip evidence with no fallback.

## Out of Scope

- Scaling benchmarks, launch-overhead claims, profiling, or memory budgeting.
- New physics, tolerance relaxation, exact stochastic seed replay across devices,
  dynamic resizing, compaction, migration, or CPU fallback.
- Public graph-capture exports, automatic recapture, checkpointing graph handles,
  rollback after a writer launch, or changes to E8-F1--E8-F4 contracts.
