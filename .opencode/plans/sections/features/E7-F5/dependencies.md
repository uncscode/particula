# Dependencies

## Upstream

- **E7-F2 / T2 — Backend-selected condensation:** required adapter,
  thermodynamic configuration, coupled gas semantics, and parity contract.
- **E7-F3 / T3 — Backend-selected Brownian coagulation:** required adapter,
  collision outputs, and persistent RNG-resource contract.
- **E7-F4 / T4 — GPU-resident session:** required lifecycle, fixed-shape
  particle/gas/environment state, resource registry, checkpoint boundary, and
  post-launch fault behavior.
- **Inherited E7-F1 and E7-F6:** typed execution context plus capability,
  availability, explicit fallback, error, and public-export policy.
- Shipped direct GPU dilution, wall loss, and nucleation contracts and E6-F9
  five-process fixtures.

## Downstream

- **E7-F7 / T7** extends scheduler nodes with prescribed transport, mixing,
  advection, and volume evolution.
- **E7-F8 / T8** specializes persistent per-box stream identity, reset, and
  checkpoint/restart behavior across scheduled stochastic processes.
- **E7-F9 / T9** uses this timestep contract for diagnostics, complete examples,
  multi-timestep regressions, documentation, and epic closeout.
- Epics H and I depend on a stable loop but graph capture/performance and
  autodiff remain outside E7-F5.

## Phase Ordering

P1 node contracts precede P2 graph resolution. P3 process integration and P4
state updates may build on P2 but must both land before P5 can enforce freshness.
P6 composes all prior phases and validates the complete loop. P7 documents the
settled contract last. Every production phase includes co-located tests.
