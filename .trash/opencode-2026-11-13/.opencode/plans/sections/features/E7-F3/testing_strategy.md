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
- **P4 (implemented):** Focused adapter tests parameterize charged,
  sedimentation, turbulent, combined, unknown, discrete-Brownian, and inexact
  marker request-shaped values and assert exact-marker rejection before CPU
  dispatch, optional Warp import, or lazy resolver access. They cover Warp P2
  precedence (including selected time before missing RNG/alias), no-mutation
  snapshots for selection rejection, CPU execution-control rejection before its
  single runnable call, and one resolver/kernel call for native-schema and
  writer-like post-launch failures. Native exceptions propagate by identity and
  writer mutations remain visible; no recovery or rollback is asserted.
- **P5 (implemented):**
  `particula/execution/tests/coagulation_integration_test.py` adds isolated CPU
  reference and resident-Warp fixture/assertion coverage. It covers one- and
  multi-box resource identity, concentration-weighted mass and signed-charge
  conservation, active-only/disjoint pair validity, inactive-slot preservation,
  box isolation, exact zero/one-active no-merge cases, a fixed 100-trial
  three-sigma acceptance check, persistent RNG advancement, explicit RNG-reset
  replay, and optional CUDA invariant rows. The retained adapter tests prove no
  selected-path conversion, restore, implicit synchronization, or fallback.
  CPU/Warp trajectories are not compared seed-by-seed.
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
