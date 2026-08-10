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
- [x] Add the wall-loss resource/adapter path with an independently initialized,
  exact published stream view; preserve direct-kernel signatures and physics.
- [x] Thread scheduler-resolved selected box indices to the wall-loss adapter so
  disabled/prelaunch-skipped boxes do not enter a launch and cannot advance RNG.
- [x] Add explicit initialize/reset operations with session lifecycle guards;
  prohibit implicit reset from repeated seed values.
- [x] Add schema-v3 optional published-stream continuation to checkpoint/finalize:
  capture immutable metadata/current words after one synchronization, exclude RNG
  roles from ordinary payloads, and retain v1/v2 compatibility.
- [x] Restore v3 continuation through a private registry seam into fresh
  exact-device arrays/bindings without normal acquisition or reseeding; reject
  malformed records before setup/allocation and clean up failed fresh restarts.
- [x] Preserve the existing writer-capable wall-loss failure path: close the
  token, fault the session, propagate the error, and make no reset/rollback
  promise after launch.
- [x] Keep public exports limited to stable execution-layer types; retain concrete
  kernel initialization helpers in their current narrow modules.

## Tooling / Tests

- [x] Add `particula/execution/tests/rng_test.py` for P1 derivation, validation,
  canonical registry lookup, initializer preflight, host-only import, and
  known-answer vectors.
- [x] Add resident coagulation adapter/resource/session/checkpoint tests for
  sidecar identity, one-time initialization, rejection preservation, and forced
  false initialization dispatch.
- [x] Add wall-loss resource/adapter/scheduler/checkpoint regressions for
  canonical initialization, identity/nonaliasing, selected-box gating,
  disabled/zero/no-work preservation, rejection, and fault lifecycle.
- [x] Add P4 RNG/resource/session regressions for frozen manifests, selected
  reset preflight and preservation, published-only scope, exact lifecycle
  binding, empty selections, and direct-only export denial.
- [x] Add `particula/execution/tests/rng_invariance_test.py` for unrelated box
  insertion, disablement/no-work, removal, and physical-lane permutation across
  resident Brownian coagulation and neutral wall loss.
- [x] Extend `particula/execution/tests/checkpoint_test.py` with exact
  uninterrupted-versus-restart stream and stochastic-output comparisons.
- [x] Add transfer/synchronization spies proving normal steps and reset-free
  execution perform no hidden bulk transfer or host synchronization.
- [x] Run focused execution, RNG, lint, type, and strict documentation validation
  for the shipped continuation contract.
