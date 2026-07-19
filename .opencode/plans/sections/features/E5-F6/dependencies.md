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
  singleton, two-way, and four-way capability matrix for remaining
  release/cross-mechanism evidence.
- **E5-F9 — Support Documentation, Example, and Roadmap Closeout:** consumes
  supported combination names, required inputs, and single-pass limits.
- Future GPU-resident simulation work depends on E5-F6's stable combined
  execution contract; no high-level integration is part of this feature.

## Phase Ordering

P1 froze the matrix and preflight contract. P2 implemented and proved summed
rates/majorants; P3 shipped end-to-end shared-path execution; and P4 documents
the verified rows. E5-F7 remains the release/cross-mechanism validation phase,
while E5-F9 remains the consolidated example and closeout phase.

The frozen matrix contains the four singleton rows, all six unordered two-way
rows, and the full four-way row. Three-way masks are unsupported and must fail
preflight before allocation, mutation, or RNG advancement; they are not an
implicit intermediate on the path to the four-way row.

Parent: E5. Classifier diagnostics: none.
