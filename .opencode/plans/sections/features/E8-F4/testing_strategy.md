# Testing Strategy

Every phase ships tests beside the execution modules using the `*_test.py`
convention. Coverage thresholds are never lowered. Hardware-independent tests
use an injected fake capture runtime; Warp CPU is the installed-Warp uncaptured
baseline; native graph capture evidence is CUDA-only and pass-or-clean-skip.

## Per-Phase Approach

- **P1 — delivered qualification controller:**
  `particula/execution/tests/graph_capture_test.py` covers exact types and
  identities, READY/ACTIVE/closed-guard preconditions, ordered lazy adapter
  probes, malformed adapter metadata, error propagation, and the negative
  native-handle/cleanup/token/callable-invocation contract. Export denial is
  covered in `particula/execution/tests/exports_test.py`. These host-only tests
  do not claim native CUDA capture; that remains P2.
- **P2 — delivered fixed capture:** `graph_capture_test.py` covers begin →
  twelve-operation dispatch → end ordering, opaque-handle identity retention,
  pre-begin and post-end revalidation, delayed CAPTURED publication, and
  begin/dispatch/end/release failure chains. Capture-window spies deny normal
  scheduler/token work, validation, resource work, allocation, transfer,
  readback, and synchronization. `full_loop_test.py` traces the private
  dispatcher order, while `exports_test.py` denies the new carrier and capture
  entry point from package and top-level exports. A `warp`/`cuda` row uses the
  native capture APIs and twelve device no-ops; it skips only for absent Warp,
  CUDA, or required capture APIs.
- **P3 — delivered guarded replay:**
  `particula/execution/tests/graph_capture_test.py` covers authentic P2-issued
  opaque-handle provenance, exact record/binding/lifecycle/device/duration
  validation, repeated one-token/one-launch success, mutable pinned payload and
  RNG-word compatibility, and no launch after rejected preflight. It also covers
  native launch/completion failures and writer-capable no-rollback faulting.
- **P4 — delivered invalidation:** `graph_capture_test.py`,
  `gpu_session_test.py`, `checkpoint_test.py`, and `gpu_resources_test.py` use
  deterministic fake-native traces for structural drift, writer fault,
  finalization, close/discard, retirement, and recapture. They assert
  unregister-before-release, exactly one release even when release raises,
  stale provenance before token entry/launch, first-reason precedence, exact
  attachment/context checks, read-only preservation, and guarded stream-writer
  fault propagation. Checkpoint tests also cover notification ordering and no
  cached finalized checkpoint after release failure.
- **P5 — full loop:** In `captured_full_loop_test.py`, run identical fixtures for
  CPU reference, uncaptured Warp, and captured CUDA over multiple timesteps.
  Compare fields separately with documented deterministic tolerances; assert
  particle-plus-gas conservation independently; evaluate stochastic processes
  with aggregate/sigma bounds rather than exact cross-device RNG replay.

## Required Scenarios

- Single-box and multi-box configurations; empty/inactive fixed slots; stable
  closed GAS and PARTICLES communication maps; optional volume evolution and
  diagnostics.
- Successful capture, repeated replay, explicit RNG initialization before
  capture, state advancement without reseeding, and explicit reset followed by
  fresh capture where required by the E8-F1 signature contract.
- Shape, device, primary-array, sidecar, prepared-plan, schedule, process
  configuration, map, diagnostics, and graph-handle drift.
- Capture begin, enqueue, capture end, launch, token cleanup, and teardown
  failures, including operation plus cleanup failures.
- CPU capture rejection, absent CUDA, unavailable capture APIs, and exact known
  capability skips; unexpected runtime errors must fail rather than skip.

## Commands and Coverage Evidence

Focused fix checks are assertion evidence only and run coverage-disabled:

```bash
pytest particula/execution/tests/graph_capture_test.py -q --no-cov
pytest particula/execution/tests/graph_capture_test.py \
  particula/execution/tests/gpu_session_test.py \
  particula/execution/tests/checkpoint_test.py \
  particula/execution/tests/gpu_resources_test.py -q --no-cov
pytest particula/execution/tests/graph_capture_test.py -q --no-cov \
  -m "warp and cuda"
pytest particula/execution/tests/full_loop_test.py -q --no-cov
pytest particula/execution/tests/exports_test.py -q --no-cov
```

The CUDA command may pass or cleanly skip; it never falls back to CPU. A focused
target with `--cov` is invalid comprehensive coverage evidence and must not be
classified as a fix failure merely because it cannot meet full-package
coverage. After focused assertions pass, run the untargeted repository runner,
which supplies repository-configured full-package coverage and its normal
threshold:

```bash
.opencode/tools/run_pytest.py
```

If graph work changes resident lifecycle modules covered by the closeout policy,
also retain per-target term-missing rows and the aggregate 80% gate for the
actual executable-module diff. Documentation changes require
`mkdocs build --strict`.

## Coverage Impact

The new concrete graph owner requires branch coverage for capability, cleanup,
signature comparison, lifecycle, replay, and fault paths. Fake-runtime unit
tests provide deterministic coverage without CUDA; CUDA rows prove native
integration but are not relied on to satisfy the repository threshold.
