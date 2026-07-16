# Dependencies

## Upstream

- **E5-F1 — Mechanism Configuration and Sampling Contract:** supplies canonical
  masks, structural/executable validation, additive dispatch, and single-pass
  sampler extension rules.
- **E5-F2 — Charged Pair Physics and Charge-Conserving Merges:** supplies
  approved charge physics and the merge path that transfers and clears charge.
- **E5-F3 — Charged and Brownian-Plus-Charged Execution:** supplies executable
  charged dispatch, a proven charged majorant, and the first additive two-term
  precedent.
- **E5-F4 — SP2016 Sedimentation GPU Execution:** supplies efficiency-1 pair
  rates, particle properties, and a safe sedimentation majorant.
- **E5-F5 — ST1956 Turbulent-Shear GPU Execution:** supplies explicit
  dissipation/fluid-density input validation, pair rates, and a safe shear
  majorant.
- Shipped E2/E3 GPU data, fixed-buffer, bounded sampling, and persistent RNG
  ownership contracts; CPU `CombineCoagulationStrategy` additive semantics;
  NVIDIA Warp for required Warp CPU evidence.

## Downstream

- **E5-F7 — Cross-Mechanism Validation Matrix:** consumes the finalized
  additive capability matrix and two-/four-way fixtures for release evidence.
- **E5-F9 — Support Documentation, Example, and Roadmap Closeout:** consumes
  supported combination names, required inputs, and single-pass limits.
- Future GPU-resident simulation work depends on E5-F6's stable combined
  execution contract; no high-level integration is part of this feature.

## Phase Ordering

P1 freezes the executable matrix and preflight contract. P2 consumes that mask
to implement and prove summed rates/majorants. P3 validates end-to-end one-pass
execution only after P2's deterministic bound proof. P4 documents only shipped,
verified rows. E5-F7 starts after P3; E5-F9 consumes P4 and E5-F7 results.

Parent: E5. Classifier diagnostics: none.
