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
- **P4 (implemented in issue #1487):**
  `particula/execution/tests/gpu_session_test.py` covers zero, float, and
  `Rational` step cycles; opaque token identity/frozen mutation; invalid
  duration; nested, missing, fabricated, cross-guard, mismatched, and repeated
  completion; terminal lifecycle and primary-identity drift; and
  `assert_step_closed()` future-boundary stubs. It also proves external adapter
  failure leaves the token open and counters unchanged. Conversion, restore,
  synchronization, upload, and allocation spies establish that guard paths do
  none of those operations.
  `particula/execution/tests/gpu_resources_test.py` covers
  `validate_pinned_session()` for the exact session, distinct sessions,
  fabricated terminal states, and primary/container drift, including unchanged
  bindings/views and no allocation or mutation.
- **P5 (implemented in issue #1488):**
  `particula/execution/tests/checkpoint_test.py` covers frozen immutable payload
  records, detached inspection carriers, canonical primary/vapor-pressure
  restoration, active checkpoint behavior, cached terminal finalization, fresh
  restart identities, acquired resource-family restoration, malformed payload
  rejection before setup, and the closed-step rejection path. Warp cases defer
  import and use the Warp CPU baseline. Transfer/order, failure-injection,
  descriptor-matrix, and optional CUDA coverage are retained as focused
  regression expectations for this concrete-only boundary.
- **P6 (implemented in issue #1489):**
  `particula/execution/tests/gpu_session_test.py` injects before-token,
  read-only-after-token, and writer-may-have-launched failures. It asserts exact
  original exception/traceback propagation, token release without guard
  counter/time advancement, reusable read-only sessions, observable writer
  mutation with `FAULTED` lifecycle, and deterministic pre-runtime faulted
  operation rejection. It also covers invalid failure-seam/abort inputs and
  cleanup failures without masking the original error. Close/discard rows cover
  active, faulted, closed, and finalized behavior; binding/open-token rejection;
  active-validation-once versus faulted identity-only validation; retained
  payload/checkpoint identity; and spies proving no synchronization, conversion,
  checkpoint/restart, allocation, restore, retry, migration, or fallback.
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
pytest particula/execution/tests/gpu_session_test.py particula/execution/tests/gpu_resources_test.py --cov=particula.execution.gpu_session --cov=particula.execution.gpu_resources --cov-report=term-missing --cov-fail-under=80 -q -Werror
pytest particula/execution/tests/checkpoint_test.py -q -Werror
pytest particula/execution/tests/checkpoint_test.py --cov=particula.execution.checkpoint --cov-report=term-missing --cov-fail-under=80 -q -Werror
pytest particula/gpu/tests/process_sequence_test.py -q -Werror
pytest particula/gpu/tests/kernel_exports_test.py -q -Werror
mkdocs build --strict
```
