# Dependencies

## Upstream

- Parent epic E3 exists and frames this feature as GPU kernel correctness and
  low-level API hardening.
- No child feature dependency blocks E3-F1. Classifier diagnostics reported none.
- Existing Warp coagulation kernel infrastructure, RNG initializer, and
  co-located test fixtures must remain available.

## Downstream

- Sibling E3 feature tracks can reuse the same API-hardening practices: explicit
  buffer ownership, validation before mutation, and CPU/CUDA-if-available tests.
- GPU benchmarks and graph-capture-oriented workflows depend on a stable
  seed-once RNG contract.
- Documentation for GPU data containers and roadmap correctness depends on the
  final semantics described by this feature.

## Phase Ordering Notes

- P1 must happen first because implementation depends on the compatibility
  contract for `rng_seed`, `rng_states`, and initialization mode.
- P2 can be developed immediately after P1 and should fail against the current
  unconditional reinitialization behavior.
- P3 implements the behavior needed to satisfy P1/P2 tests.
- P4 ships last so documentation reflects the final API and observed test
  behavior.
