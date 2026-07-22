# Open Questions

All E6-F3 planning questions were resolved on 2026-07-21 from the CPU wall-loss
reference and existing direct-Warp process contracts.

- [x] Which neutral transport primitives are reused?
  - Decision: reuse existing viscosity, mean-free-path, Knudsen, slip,
    mobility, diffusion, radius, density, and settling functions at their
    current export levels. Add only concrete-module Debye/coth-safe and
    wall-loss coefficient helpers; do not broaden unrelated public exports.
- [x] What does `time_step == 0` do to RNG state?
  - Decision: after validation it performs no draws and no particle writes.
    Supplied RNG is byte-for-byte unchanged unless the caller explicitly asks
    for `initialize_rng=True`, whose reset remains an intentional side effect.
- [x] Which fixed-slot active predicate applies?
  - Decision: use the E6 shared truth table. Positive finite concentration and
    positive finite total mass are active; the all-zero mass/concentration/
    charge record is free; every contradictory half-active record raises before
    RNG or mutation. Legacy inactive sentinels outside new E6 APIs are not
    retrofitted here.
- [x] How is wall-loss RNG initialized?
  - Decision: extract a private shared per-box `wp.uint32` initialization
    primitive and expose a wall-loss-specific concrete-module wrapper. Wall loss
    and coagulation retain separate caller-owned streams; no generic shared
    public stream is introduced.
- [x] Are deterministic coefficients returned by the public step?
  - Decision: no. Return the same particle container and test coefficient
    parity through concrete-module helpers. Add a caller-owned diagnostic only
    if a later integration demonstrates a stable user need.
- [x] Must CPU and Warp use identical random-number sequences?
  - Decision: no. Require deterministic coefficient parity and statistically
    bounded outcomes, not exact RNG sequence parity.
- [x] Does E6-F3 include charged loss, slot management, a high-level GPU
  runnable, or backend selection?
  - Decision: no. Those remain owned by E6-F4, E6-F5/F6, and Epic G.
