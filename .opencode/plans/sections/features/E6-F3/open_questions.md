# Open Questions

All E6-F3 planning questions were resolved on 2026-07-21 from the CPU wall-loss
reference and existing direct-Warp process contracts.

- [x] Which neutral transport primitives are reused?
  - Decision (implemented in P1 / #1401): `particula.gpu.properties` is the
    canonical owner/import surface for neutral particle radius, slip, diffusion,
    effective density, and settling helpers. Gas viscosity and mean free path
    remain in `gas_properties.py`. Legacy `particula.gpu.dynamics` definitions
    and re-exports were removed rather than retained as compatibility wrappers.
    Device-only `debye_1_wp` and `x_coth_x_wp` were added in particle properties;
    no wall-loss coefficient or high-level API export was broadened.
- [x] What does `time_step == 0` do to RNG state?
  - Decision (implemented in P5 / #1405): after validation it performs no draws,
    no particle writes, and no supplied-sidecar initialization or reset, even
    when `initialize_rng=True`.
- [x] Which fixed-slot active predicate applies?
  - Decision: use the E6 shared truth table. Positive finite concentration and
    positive finite total mass are active; the all-zero mass/concentration/
    charge record is free; every contradictory half-active record raises before
    RNG or mutation. Legacy inactive sentinels outside new E6 APIs are not
    retrofitted here.
- [x] How is wall-loss RNG initialized?
  - Decision (implemented in P5 / #1405): a private per-box `wp.uint32`
    initializer seeds omitted private state for each successful positive-time
    call and resets supplied state only with `initialize_rng=True`. Wall loss and
    coagulation retain separate caller-owned streams; no generic public stream
    is introduced.
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

## P4-P5 Resolutions (#1404, #1405)

- [x] Does a valid positive-time P4 call mutate slots?
  - Decision: yes. Following frozen P3 preflight it evaluates neutral
    coefficients for usable slots and clears all mass lanes, concentration, and
    charge for stochastic removals. Zero time is the only execution no-op.
- [x] Does the shipped direct step own caller RNG lifecycle?
  - Decision (implemented in P5 / #1405): supplied `rng_states` remains
    caller-owned and mutates in place only after successful positive-time
    preflight. One sequential owner advances each per-box state for eligible
    fixed slots only; an omitted sidecar is private, and repeated `rng_seed`
    values do not reseed supplied state without `initialize_rng=True`.
