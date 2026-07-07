# E2-F8 Open Questions

## Resolved Answers

1. P2 chose explicit runtime rejection for CPU coagulation `ParticleData`
   inputs with `n_boxes != 1` by adding a reusable shared guard in
   `coagulation_strategy_abc.py` before helper reads and `step()` mutation.
2. Representative condensation public multi-box rejection coverage now includes
   both the existing `mass_transfer_rate(...)` path and a public `step()` path
   that flows through `_require_single_box`.
3. No general documentation update was needed in P2; broader caller guidance and
   user-facing support-boundary wording remain deferred to the later doc phase.
4. The tested CPU dynamics boundary now matches the intended E2-F2-aligned
   wording: multi-box containers may exist, but these CPU strategy paths are
   still single-box-only unless documented otherwise.

## Deferred Out-of-Scope Topics

- Full multi-box CPU condensation or coagulation implementation details.
- First-class all-box execution semantics for GPU strategy kernels.
- Container schema changes for representing environment boxes differently.
