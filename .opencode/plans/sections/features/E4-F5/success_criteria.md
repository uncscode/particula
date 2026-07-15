# Success Criteria

- [x] Issue #1302: `partitioning == 0` leaves particle species strictly
  unchanged; `gas.concentration` remains unchanged for the particle-only path.
- [x] Issue #1303 direct helper: finalized uptake never exceeds current gas
  plus same-box/species release from the already-gated proposal.
- [x] Issue #1303 direct helper: evaporation never exceeds per-particle owned
  mass.
- [x] Issue #1304: particle gain equals gas loss for every tested box/species at
  tight deterministic bookkeeping tolerance.
- [x] Issue #1304: gas and particle inventories remain finite/nonnegative and
  inactive slots remain unchanged in focused Warp regressions.
- [x] Issue #1304: all four fixed substeps use current, already-coupled gas for
  subsequent mass-transfer proposals.
- [x] Issue #1304: returned transfer and E4-F4 heat use exact finalized applied
  transfer.
- [x] Issue #1302: partitioning and supplied P2-sidecar
  shape/device/dtype/value failures occur before observable state mutation.
- [x] Issue #1302: binary-mask checking uses only a private status readback; no
  container/data transfer, schema change, or caller-scratch allocation occurs.
- [x] Issue #1303: invalid direct-helper proposal/P2-sidecar inputs fail before
  masses, gas, proposal, or supplied P2 sidecars change.
- [x] Issue #1303: the public P1 path launches no P2 inventory kernels and
  leaves supplied P2 sidecars untouched.
- [x] Issue #1305: production-hook and issue #1272 integration conservation
  regressions land together, with H2O/NH4HSO4 accounting kept independent and
  gas-only N2 exactly invariant.
- [x] Issue #1305: deterministic fp64 public-hook bookkeeping is conserved per
   box/species at `rtol=1e-12, atol=1e-30`, separately from CPU-oracle particle/
   gas parity at `rtol=2e-10, atol=1e-30`; Warp CPU passes and CUDA is guarded.
- [x] Issue #1306: the roadmap and foundation guide describe only the verified
   P1-P4 bounded direct-kernel contract and retain E4-F6/E4-F7 as gates for
   broader GPU-condensation and E4 production support.

| Metric | Baseline | Target | Source |
|---|---|---|---|
| Per-box/species conservation residual | Not guaranteed | `rtol=1e-12`, `atol=1e-30` | CPU integration and GPU public-hook regression |
| Negative gas/particle inventories | Possible after independent clamps | 0 cases | GPU unit/integration tests |
| Disabled-species mutations | Gate ignored | 0 | Partitioning tests |
| Coupled substeps using current gas | 0 | 4 of 4 | Substep regression |
| Hidden CPU/GPU transfers | Not permitted | 0 | Transfer-boundary tests |
| Production hook paired with regression | Missing | Shipped in #1305 | Issue #1272 gate |
| Bounded support documentation | Deferred | Shipped in #1306 | Roadmap and foundation guide |
