# Success Criteria

- [ ] Fixed-shape communication declarations reject malformed indices, shapes,
  devices, aliases, values, outbound demand, and unsupported modes before commit.
- [ ] Empty or disabled maps preserve independent-box behavior and perform no
  communication write after validation.
- [ ] Prescribed positive per-box volume changes update `particles.volume` by
  identity and conserve particle/gas extensive inventory through concentration
  renormalization.
- [ ] Gas advection and simple mixing are synchronous, edge-order independent,
  and match an independent NumPy reference.
- [ ] Particle transport preserves population, species mass, and charge for
  closed maps and fails without partial state commit when fixed capacity cannot
  represent a prescribed transfer.
- [ ] Open boundaries expose explicit source/sink accounting; no loss or source
  is silently attributed to numerical transport.
- [ ] E7-F4 resources remain fixed-shape and identity-stable across repeated
  steps and checkpoint/restart.
- [ ] E7-F5 executes communication/volume nodes at the documented canonical
  barrier and refreshes invalidated derived state before consuming processes.
- [ ] E7-F6 capability errors and explicit fallback boundary are preserved; no
  runtime retry, hidden transfer, implicit sync, or silent fallback occurs.
- [ ] Warp CPU parity, equivalent one-box, isolated-box metamorphic, 1D
  advection/mixing, expansion, and repeated-step tests pass with explicit
  tolerances; optional CUDA rows skip cleanly when unavailable.
- [ ] Changed modules retain at least 80% coverage, repository thresholds are not
  lowered, and strict documentation validation passes.
- [ ] Full CFD, dynamic capacity, graph capture/performance, autodiff, E7-F8 RNG,
  and E7-F9 publication work remain outside the delivered claim.

## Metrics

| Metric | Baseline | Target | Source |
|---|---:|---:|---|
| Closed-map gas amount error | No integrated transport | `rtol <= 1e-12` or documented stricter case tolerance | Independent NumPy ledger tests |
| Closed-map particle species-mass error | No integrated transport | `rtol <= 1e-12`, scale-appropriate `atol` | Multi-box parity tests |
| Normal-step bulk CPU/GPU transfers | Not available | 0 between explicit checkpoints | Transfer spies |
| Normal-step explicit synchronizations/readbacks | Not available | 0 | Scheduler/session spies |
| Duplicate runtime allocations after setup | Not available | 0 for registered communication scratch | Identity/allocation tests |
| Edge-order dependent fixtures | Not available | 0 | Permutation tests |
| Partial commits on precommit rejection | Not available | 0 | State snapshots |
