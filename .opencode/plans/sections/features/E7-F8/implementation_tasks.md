# Implementation Tasks

## Execution and Backend Integration

- [x] Add typed stream identity, descriptor, derivation-version, and registry
  models in `particula/execution/rng.py`.
- [x] Specify deterministic non-`hash()` root-seed/process/logical-box mixing and
  freeze known-answer vectors.
- [x] Validate unique stable logical box IDs, process namespaces, seed range,
  dimensions, state dtype/device, and registry completeness before mutation.
- [x] Add immutable P1-derived resident stream metadata and one resident-owned
  coagulation state array to the exact `ResidentSession`/resource binding;
  initialize it only on first acquisition.
- [x] Update the E7-F3 Brownian resident adapter/scheduler path to require the
  pinned coagulation view and pass `initialize_rng=False` during every step.
- [ ] Add or update the wall-loss adapter to consume its separate stream view and
  preserve existing direct-kernel signatures and mutation behavior.
- [ ] Extend E7-F5 resolved scheduling so disabled process/box work does not
  advance a stream while enabled boxes retain stable logical identity.
- [ ] Add explicit initialize/reset operations with session lifecycle guards;
  prohibit implicit reset from repeated seed values.
- [x] Fail closed before checkpoint/finalize payload work when a resident
  coagulation sidecar is published; do not serialize descriptors/state or support
  restart continuation in P2.
- [ ] Fault the session after uncertain post-launch failures and prohibit reset,
  checkpoint, or continued stepping from uncertain state.
- [ ] Keep public exports limited to stable execution-layer types; retain concrete
  kernel initialization helpers in their current narrow modules.

## Tooling / Tests

- [x] Add `particula/execution/tests/rng_test.py` for P1 derivation, validation,
  canonical registry lookup, initializer preflight, host-only import, and
  known-answer vectors.
- [x] Add resident coagulation adapter/resource/session/checkpoint tests for
  sidecar identity, one-time initialization, rejection preservation, and forced
  false initialization dispatch.
- [ ] Add `particula/execution/tests/rng_invariance_test.py` for unrelated box
  insertion, disablement, removal, and permutation.
- [ ] Extend `particula/execution/tests/checkpoint_test.py` with exact
  uninterrupted-versus-restart stream and stochastic-output comparisons.
- [ ] Add transfer/synchronization spies proving normal steps and reset-free
  execution perform no hidden bulk transfer or host synchronization.
- [ ] Run Warp CPU as required baseline, optional CUDA parametrization, export
  tests, focused coverage at or above 80%, and strict documentation validation.
