# Dependencies

## Upstream

- **E6-F1 / T1 (required):** CPU dilution strategy and runnable reference. E6-F2
  must consume its canonical finite-step equation, validation semantics, units,
  deterministic fixtures, and protected-field/no-op expectations. GPU work must
  not freeze conflicting behavior before T1 is available.
- Existing `WarpParticleData`, `WarpGasData`, explicit conversion helpers, and
  direct-kernel validation conventions from shipped GPU foundations.
- Warp is required for implementation and Warp CPU test evidence; CUDA is an
  optional validation target and cannot be a delivery blocker.

## Downstream

- **E6-F9 / T9:** integrated direct-call validation and explicit-transfer
  checkpoint example consume the direct dilution entry point.
- Epic G backend selection and resident scheduling may later orchestrate this
  step, but are explicitly outside E6-F2 and remain downstream of Epic F.
- E6-F3 through E6-F8 do not need to block this feature except through shared
  container/API stability; they must not be coupled into dilution kernels.

## Phase Ordering

P1 freezes the T1-derived contract before P2 implements kernels. P3 completes
atomic validation before P4 declares parity. P5 documents only the behavior and
evidence shipped by P1-P4. Unit tests are co-located with each implementation
phase; there is no standalone testing phase.
