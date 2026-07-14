# Success Criteria

- [x] Issue #1302: `partitioning == 0` leaves particle species strictly
  unchanged; `gas.concentration` remains unchanged for the particle-only path.
- [ ] Finalized uptake never exceeds current gas plus same-substep evaporation.
- [ ] Evaporation never exceeds per-particle or aggregate particle inventory.
- [ ] Particle gain equals gas loss for every box/species at tight tolerance.
- [ ] Gas and particle inventories remain finite and nonnegative; inactive slots are unchanged.
- [ ] All four fixed substeps consume current, already-updated gas state.
- [ ] Returned transfer and E4-F4 heat use the exact finalized applied transfer.
- [x] Issue #1302: partitioning and supplied P2-sidecar
  shape/device/dtype/value failures occur before observable state mutation.
- [x] Issue #1302: binary-mask checking uses only a private status readback; no
  container/data transfer, schema change, or caller-scratch allocation occurs.
- [ ] Production hook and issue #1272 conservation regression land together.
- [ ] CPU reference and Warp CPU pass; CUDA parity passes when available.

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Per-box/species conservation residual | Not guaranteed | Within explicit tight fp64 bookkeeping tolerance | Integration regression |
| Negative gas/particle inventories | Possible after independent clamps | 0 cases | GPU unit/integration tests |
| Disabled-species mutations | Gate ignored | 0 | Partitioning tests |
| Coupled substeps using current gas | 0 | 4 of 4 | Substep regression |
| Hidden CPU/GPU transfers | Not permitted | 0 | Transfer-boundary tests |
| Production hook paired with regression | Missing | Same issue/change | Issue #1272 gate |
