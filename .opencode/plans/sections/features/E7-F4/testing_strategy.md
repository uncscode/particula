# Testing Strategy

Coverage thresholds must never be lowered. Every production phase ships its
unit tests in the same change, test files use the `*_test.py` convention, and
changed execution modules must retain at least 80% coverage. Warp CPU is the
deterministic baseline; CUDA rows are optional and skip cleanly.

## Per-Phase Coverage

- **P1 (implemented in issue #1484):**
  `particula/execution/tests/gpu_session_test.py` covers frozen dimensions and
  metadata, all four immutable lifecycle values (not transitions), concrete-only
  exports, lazy no-Warp imports, missing-Warp ordering/error text, identity
  retention, zero particle/species shapes, generated-container form, and
  isolated primary-array dtype/shape/device rejection. Metadata-operation
  sentinels verify successful and invalid-schema construction performs no host
  payload read, synchronization, launch, or allocation. Warp-dependent tests
  are marked `warp` and import Warp inside test bodies.
- **P2 (implemented in issue #1485):**
  `particula/execution/tests/gpu_session_test.py` uses a subprocess matrix that
  blocks Warp/GPU imports to prove exact-device, carrier, rank/shape, and
  gas-name failures occur locally. Warp-CPU conversion spies prove one call per
  carrier in particle/gas/environment order with the unchanged native device;
  tests also cover input identity/immutability, failures at every conversion
  position, and final `ResidentSession` schema failure. E7-F6 availability
  rejection remains untested because its public runtime seam is absent; P2
  neither probes nor falls back.
- **P3:** Parameterize all canonical sidecar shapes and dtypes in
  `particula/execution/tests/gpu_resources_test.py`. Assert same-device
  ownership, exact identity across repeated acquisition, duplicate/unknown key
  rejection, prohibited aliases, and no broad concrete-record exports.
- **P4:** Run multiple empty scheduler-facing lifecycle cycles and process
  adapter stubs while a generalized conversion guard proves zero
  `from_warp_*` calls and zero session-level synchronization. Assert dimensions,
  container identities, and resource identities remain stable.
- **P5:** In `particula/execution/tests/checkpoint_test.py`, assert one explicit
  sync and three `sync=False` restores, ordered-name retention, gas
  vapor-pressure lossiness is represented in metadata/resources, active state
  remains usable after checkpoint, finalization is terminal, and restart
  matches an uninterrupted deterministic Warp CPU sequence.
- **P6:** Inject validation failures, allocation failures, failures before
  process launch, and simulated post-launch failures. Assert no rollback or
  fallback, faulted-state guards, original error propagation, no implicit
  restore on close, and idempotent close/finalize rules.
- **P7:** Run `mkdocs build --strict`, link validation, no-Warp import tests, and
  focused documented examples.

## Integration and Regression Checks

- Reuse `particula/gpu/tests/process_sequence_test.py` one-box and multi-box
  fixtures and its intermediate-restore guard.
- Preserve every direct-kernel test suite; the session changes orchestration
  and ownership only, not physics.
- Compare checkpoint/restored CPU arrays with exact equality when no physics
  occurs and existing explicit tolerances/conservation checks when direct steps
  occur.
- Assert no bulk `to_warp_*` or `from_warp_*` calls between setup and explicit
  checkpoint/finalize boundaries.
- Assert stable object identities and shapes across at least two timesteps and
  across a nonterminal checkpoint.
- Keep stochastic checks scoped to state preservation. E7-F8 owns per-box
  stream invariance and exact restart policy; CPU/CUDA trajectory equality is
  not required.

## Focused Commands

```bash
pytest particula/execution/tests/gpu_session_test.py -q -Werror
pytest particula/execution/tests/gpu_resources_test.py -q -Werror
pytest particula/execution/tests/checkpoint_test.py -q -Werror
pytest particula/gpu/tests/process_sequence_test.py -q -Werror
pytest particula/gpu/tests/kernel_exports_test.py -q -Werror
mkdocs build --strict
```
