# Dependencies

## Upstream

- E5-F1, Mechanism Configuration and Sampling Contract, must freeze stable
  mechanism identifiers, the approved charged-model subset, particle-resolved
  support, and the pair-rate extension contract before E5-F2-P2 is finalized.
- Shipped E2 foundations provide `WarpParticleData.charge`, fp64 fixed-shape
  arrays, and explicit CPU/Warp conversion boundaries.
- Shipped E3 coagulation hardening provides disjoint bounded pair selection,
  fail-before-launch validation, persistent RNG state, and caller-owned buffers.
- CPU formula references in `charged_dimensional_kernel.py`,
  `charged_dimensionless_kernel.py`, `charged_kernel_strategy.py`, Coulomb
  property modules, and CPU charge-conserving merge semantics.
- NVIDIA Warp. Warp CPU is required when installed; CUDA is optional evidence.

## Downstream

- E5-F3 depends on these pair helpers and charge-safe merge semantics to add a
  charged majorant and charged/Brownian-plus-charged one-pass execution.
- E5-F6 will add approved charged rates to total pair rates and majorants for
  additive multi-mechanism execution.
- E5-F7 consumes deterministic pair parity and conservation fixtures in the
  cross-mechanism validation matrix.
- E5-F9 documents the final supported charged variants and direct execution
  flow only after E5-F3/F7 establish executable evidence.

## Phase Ordering

P1 establishes shared pair properties before P2 composes model-specific rates.
P2 is gated by the E5-F1/F2 model decision. P3 makes charge input safe before P4
mutates it. P4 must ship before E5-F3 can expose charged execution. P5 documents
only tested behavior and remains last. Tests are co-located with P1-P4 rather
than deferred to a standalone testing phase. Classifier diagnostics: none.
