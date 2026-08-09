# Success Criteria

- [x] P1 fixed-shape communication declarations reject malformed metadata,
  indices, schemas, devices, aliases, domains, topology, and duplicate directed
  edges through a read-only boundary (#1507). Population-dependent outbound
  demand remains a P3 criterion because P1 has no inventory or time-step input.
- [x] P3 (#1509) empty/disabled and zero-time maps preserve independent-box
  state and caller-owned work/accounting storage after full validation.
- [x] P2 (#1508) accepts validated positive same-device final `(B,)`
  `wp.float64` volumes, updates `particles.volume` by identity, and conserves
  particle/gas extensive inventory through `old_volume / final_volume`
  concentration renormalization. It rejects invalid schema/alias/domain or
  unsafe scale inputs before mutation, and equal-volume calls are write-free.
- [x] P3 (#1509) gas advection/mixing is synchronous from immutable
  concentration-times-volume ledgers, edge-order independent, and covered by
  an independent NumPy oracle.
- [x] P4 (#1510) particle transport preserves population, species mass, and
  signed charge for closed maps using immutable pre-step exact matching or
  free-slot reservations, and gates its one-kernel commit on a representable
  fixed-capacity plan.
- [x] P3 (#1509) open boundaries expose caller-owned source/sink amount ledgers;
  no loss or source is silently attributed to numerical transport.
- [x] P5 (#1511) pins one fixed-shape, nonaliasing closed-map communication
   resource family by identity across repeated resident steps and recreates
   fresh resources on schema-v2 restart; schema-v1 noncommunication restart is
   retained.
- [x] P5 (#1511) executes communication then volume evolution as the canonical
   twelve-node pre-process barrier. Both invalidate saturation ratio only;
   existing consumer windows refresh it without unnecessarily invalidating vapor
   pressure.
- [x] P5 (#1511) preserves no runtime retry, hidden transfer, implicit sync, or
   silent fallback. Preflight failure is reusable; writer-path failure closes
   the guard and faults the session without rollback.
- [x] P6 (#1512) adds independent NumPy `float64` parity/conservation evidence
  for direct and resident multi-box communication: equivalent one-box,
  isolated-box, padded 1D/mixing, expansion/compression, sparse particle,
  complete open-ledger direct, edge-permutation, and repeated-step cases use
  immutable-prestate particle planning plus explicit
  `rtol=1e-12` and documented `atol` declarations; optional CUDA rows skip
  cleanly.
- [x] Focused documentation-contract tests and strict documentation validation
  pass; no repository coverage threshold was lowered. This documentation-only
  phase adds no production module requiring coverage evidence.
- [x] Full CFD, dynamic capacity, graph capture/performance, autodiff, E7-F8 RNG,
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
