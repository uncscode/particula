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
  - Decision: yes. The shipped example uses one box, four fixed slots, and two
    species; broader P2 fixtures retain multi-box/multi-species coverage.
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
  - Decision: no. The shipped path converts each CPU container once, invokes the
    five direct steps in order on resident state, synchronizes once, then restores
    each container once with `sync=False`.
- [x] How does the example behave without Warp or after a direct-boundary error?
  - Decision: `PARTICULA_EXAMPLE_FORCE_NO_WARP="1"` exits before any Warp probe;
    natural unavailable-Warp probing imports only `warp`. Both return stable
    no-kernel metadata. Enabled loader, conversion, direct-call, synchronization,
    and restore errors propagate unchanged; no CPU fallback or partial checkpoint
    is attempted.
- [x] Which tolerances and focused commands are published?
  - Decision: publish a per-process table, not one combined tolerance. Use exact
    equality for discrete/no-op/fallback fields; `rtol=1e-12`, `atol=1e-30`
    for mass and conservation; each process's measured parity tolerance; and
    exact-binomial wall survival bounds. P2 records focused `-Werror` process
    sequence commands for the full module, non-CUDA Warp parity, non-CUDA Warp
    stochastic, and optional CUDA selections. The example command remains P3
    work.
