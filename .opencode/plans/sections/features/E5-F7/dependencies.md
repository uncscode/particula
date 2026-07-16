# Dependencies

## Upstream

- **E5-F1 — Mechanism Configuration and Sampling Contract:** canonical mechanism
  names/masks, preflight semantics, single-pass sampler, buffer, and RNG rules.
- **E5-F2 — Charged Pair Physics and Charge-Conserving Merges:** independent
  charged formulas and transfer/clear behavior needed by charge assertions.
- **E5-F3 — Charged and Brownian-Plus-Charged Execution:** executable charged
  rows and their proven majorants.
- **E5-F4 — SP2016 Sedimentation GPU Execution:** efficiency-1 sedimentation
  properties, pair rates, majorant, and executable row.
- **E5-F5 — ST1956 Turbulent-Shear GPU Execution:** explicit dissipation and
  fluid-density inputs, pair rates, majorant, and executable row.
- **E5-F6 — Additive Multi-Mechanism Coagulation:** final approved two-way and
  four-way matrix, component-majorant sum, and one-pass execution contract.
- Shipped E2/E3 fixed-shape GPU data, device validation, bounded Brownian
  sampling, caller-owned buffers, and persistent RNG contracts.
- NVIDIA Warp for required Warp CPU evidence when installed. CUDA is optional.

## Downstream

- **E5-F9 — Support Documentation, Example, and Roadmap Closeout:** consumes the
  final evidence table, support limits, and focused reproduction commands.
- The next GPU process-completeness epic relies on E5-F7 to distinguish validated
  mechanism capability from deferred or unsupported behavior.
- Release reviewers and regression triage use this matrix as the canonical
  cross-mechanism correctness baseline.

## Phase Ordering

E5-F7 begins only after executable E5-F3 through E5-F6 rows are stable. P1
freezes the canonical matrix and independent deterministic oracles. P2 reuses P1
fixtures for end-to-end conservation and ownership evidence. P3 adds bounded
statistics only after deterministic rates and invariants are trusted. P4
publishes evidence after P1-P3 pass and then unblocks E5-F9 closeout.

Parent: E5. Classifier diagnostics: none.
