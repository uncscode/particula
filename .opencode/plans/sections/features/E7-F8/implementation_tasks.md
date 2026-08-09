# Implementation Tasks

## Execution and Backend Integration

- [x] Add typed stream identity, descriptor, derivation-version, and registry
  models in `particula/execution/rng.py`.
- [x] Specify deterministic non-`hash()` root-seed/process/logical-box mixing and
  freeze known-answer vectors.
- [x] Validate unique stable logical box IDs, process namespaces, seed range,
  dimensions, state dtype/device, and registry completeness before mutation.
- [ ] Extend E7-F4 `SidecarRegistry`/`ResidentSession` to own separate coagulation
  and wall-loss state arrays plus logical-ID-to-lane metadata.
- [ ] Update the E7-F3 Brownian adapter to require the resident coagulation view,
  initialize only at setup/reset, and pass `initialize_rng=False` during steps.
- [ ] Add or update the wall-loss adapter to consume its separate stream view and
  preserve existing direct-kernel signatures and mutation behavior.
- [ ] Extend E7-F5 resolved scheduling so disabled process/box work does not
  advance a stream while enabled boxes retain stable logical identity.
- [ ] Add explicit initialize/reset operations with session lifecycle guards;
  prohibit implicit reset from repeated seed values.
- [ ] Include stream descriptors and mutable state in the versioned E7-F4
  checkpoint and validate compatibility before fresh-session restart.
- [ ] Fault the session after uncertain post-launch failures and prohibit reset,
  checkpoint, or continued stepping from uncertain state.
- [ ] Keep public exports limited to stable execution-layer types; retain concrete
  kernel initialization helpers in their current narrow modules.

## Tooling / Tests

- [x] Add `particula/execution/tests/rng_test.py` for P1 derivation, validation,
  canonical registry lookup, initializer preflight, host-only import, and
  known-answer vectors.
- [ ] Add adapter tests for persistent coagulation and wall-loss state identity,
  initialization count, rejection preservation, and process separation.
- [ ] Add `particula/execution/tests/rng_invariance_test.py` for unrelated box
  insertion, disablement, removal, and permutation.
- [ ] Extend `particula/execution/tests/checkpoint_test.py` with exact
  uninterrupted-versus-restart stream and stochastic-output comparisons.
- [ ] Add transfer/synchronization spies proving normal steps and reset-free
  execution perform no hidden bulk transfer or host synchronization.
- [ ] Run Warp CPU as required baseline, optional CUDA parametrization, export
  tests, focused coverage at or above 80%, and strict documentation validation.
