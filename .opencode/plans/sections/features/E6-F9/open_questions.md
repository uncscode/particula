# Open Questions

All E6-F9 planning choices that can be fixed before implementation were resolved
on 2026-07-21. Measured values and command names are finalized only after their
target files exist.

- [x] Which wall-loss scenario is the smallest stable integrated fixture?
  - Decision: use neutral spherical loss with a two-call persistent RNG path and
    total survival probability near 0.5. Aggregate 4,096 initial Bernoulli
    observations across boxes and deterministic fresh-seed trials, and evaluate
    the exact family-wise binomial interval. E6-F4 retains the broader charged
    statistical matrix; E6-F9 also exercises charged mode deterministically in
    a separate integrated case.
- [x] Does the canonical example use one box while integration tests use broader
  shapes?
  - Decision: yes. Keep the example one box and one species for readability;
    use multi-box/multi-species tests for shape, isolation, and conservation.
- [x] Which exhaustion diagnostics does the example print?
  - Decision: print stable scalar summaries only: active/free counts before and
    after, requested/activated/released slots, policy code and label, scale
    factor, gas-limited events, representation-reduction events, and final-domain
    residual events (zero on success). Do not print device arrays or object
    representations.
- [x] Does E6-F9 implement backend selection or a scheduler?
  - Decision: no. Those remain owned by Epic G; E6-F9 calls direct entry points
    in one fixed validation sequence.
- [x] May the example transfer state to the host between processes?
  - Decision: no. CPU/Warp conversion occurs only at setup and final checkpoint
    boundaries.
- [x] Which tolerances and focused commands are published?
  - Decision: publish a per-process table, not one combined tolerance. Use exact
    equality for discrete/no-op/fallback fields; `rtol=1e-12`, `atol=1e-30`
    for mass and conservation; each process's measured parity tolerance; and
    exact-binomial wall survival bounds. P2 records the actual focused test and
    example commands after E6-F1 through E6-F8 create and pass those targets.
