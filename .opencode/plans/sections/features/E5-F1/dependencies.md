# Dependencies

## Upstream

- Shipped E2 GPU data foundations: `WarpParticleData`, including charge,
  fixed-shape fp64 arrays, and explicit CPU/Warp conversion boundaries.
- Shipped E3 coagulation hardening: bounded active-pair sampling, device-aware
  validation, caller-owned output buffers, and persistent RNG ownership.
- CPU additive reference semantics in `CombineCoagulationStrategy`; CPU
  mechanism implementations are formula references, not shared oracle code.
- NVIDIA Warp for required Warp CPU evidence when installed. CUDA remains
  optional and must skip cleanly when unavailable.
- Parent epic E5 constraints and issue #1320's T1 scope. Classifier diagnostics:
  none.

## Downstream

- E5-F2 uses the mechanism identifiers and extension contract while adding
  charged pair physics and charge-conserving merge behavior.
- E5-F3 registers charged and Brownian-plus-charged executable capability and
  supplies a charged majorant.
- E5-F4 and E5-F5 independently register SP2016 sedimentation and ST1956
  turbulent-shear terms after E5-F1.
- E5-F6 expands the executable combination matrix and proves safe total
  majorants for supported two- and four-way combinations.
- E5-F7 validates the completed cross-mechanism matrix; E5-F9 publishes final
  user-facing support and example material.

## Phase Ordering

P1 defines and validates configuration before P2 can consume a stable resolved
mask. P2 establishes additive dispatch before P3 wires the public entry point
and state-preserving failure tests. P4 documents only the verified contract.
E5-F2/F4/F5 should branch from the completed P1/P2 interface rather than
creating mechanism-specific sampling loops.

The initial charged identifier is `charged_hard_sphere` only. Configuration
validation must reject or retain as reserved the deferred Dyachkov, Gatti,
Gopalakrishnan, and Chahl variants before any GPU state or RNG mutation.
