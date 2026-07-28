# Testing Strategy

Every phase ships implementation and co-located `*_test.py` coverage together.
No coverage threshold may be lowered; changed execution modules target at least
80% statement coverage. Warp CPU is the installed-Warp baseline, while CUDA
rows skip cleanly when unavailable.

## Per-Phase Coverage

- **P1:** Parameterize every declared CPU/Warp Brownian capability and all
  unsupported mechanisms/distributions/devices. Assert selection rejection
  precedes adapter invocation and mutation.
- **P2 (implemented):** Focused carrier tests cover frozen CPU/Warp
  request/result identity, CPU-safe and lazy-Warp imports, ownership-form and
  metadata-detectable alias rejection, and caller-owned RNG intent retention.
  They verify construction is read-only and performs no kernel import,
  dispatch, transfer, synchronization, allocation, or RNG advancement. Native
  collision/RNG schema validation and behavioral seed/reuse/reset evidence
  remain for the future dispatch phases.
- **P3 (implemented):**
  `particula/execution/tests/coagulation_adapter_test.py` uses CPU and
  Warp-CPU spy-driven boundary tests to verify exact P3-state construction,
  local CPU control rejection, one-call CPU/Warp dispatch and forwarding,
  typed `ExecutionResult` wrapping, result identities, retained RNG intent,
  lazy resolution, and propagation of resolver/delegate/kernel failures.
  Warp tests prohibit conversion helpers, synchronization, and CPU fallback;
  they cover both direct temperature/pressure and environment-backed P2 forms.
  They do not claim native physics, stochastic, conservation, CUDA, or P4
  unsupported-mode coverage.
- **P4:** Cover malformed state, invalid time, unavailable backend/device,
  charged/sedimentation/turbulent/combined requests, output capacity mismatch,
  launch failures, and preflight-atomic versus post-launch no-rollback wording.
- **P5:** Run one-box and multi-box Brownian fixtures on Warp CPU. Compare
  deterministic rate inputs to independent CPU/NumPy references, then test
  aggregate stochastic behavior, accepted-pair bounds, mass/charge
  conservation, inactive slots, RNG progression, reset replay, and no
  intermediate transfer. Add optional CUDA rows without exact CPU/CUDA replay.
- **P6:** Execute focused examples/import snippets, documentation regression
  tests, link checks, and `mkdocs build --strict`.

## Test Locations

- New selection/adapter tests under the E7-F1-selected execution module's
  module-level `tests/` directory using `*_test.py` names.
- Reuse direct-kernel fixtures from
  `particula/gpu/kernels/tests/coagulation_test.py`,
  `coagulation_validation_test.py`, and
  `coagulation_stochastic_validation_test.py` without weakening them.
- Extend `particula/gpu/tests/kernel_exports_test.py` only for approved exports.
- Extend process/example tests only for the public selected path and explicit
  no-transfer/RNG handoff behavior.

## Acceptance Tolerances

- Conservation remains concentration-weighted and uses the strict tolerances
  established by direct-kernel fixtures where applicable.
- CPU/Warp physics comparisons record explicit `rtol`/`atol`; stochastic tests
  use aggregate or sigma-based bounds rather than seed-by-seed equality.
- Repeated Warp calls with the same persistent buffer must advance state;
  explicit reset from the same seed must replay the Warp sequence.
