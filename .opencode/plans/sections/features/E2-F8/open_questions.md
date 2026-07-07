# E2-F8 Open Questions

## Resolved Answers

1. P2 chose explicit runtime rejection for CPU coagulation `ParticleData`
   inputs with `n_boxes != 1` by adding a reusable shared guard in
   `coagulation_strategy_abc.py` before helper reads and `step()` mutation.
2. Representative condensation public multi-box rejection coverage now includes
   both the existing `mass_transfer_rate(...)` path and a public `step()` path
   that flows through `_require_single_box`.
3. The deferred doc phase is now complete: the migration guide is the canonical
   support contract, with explicit single-box guidance and caller-managed
   per-box loop documentation.
4. The tested CPU dynamics boundary now matches the intended E2-F2-aligned
   wording: multi-box containers may exist, but these CPU strategy paths are
   still single-box-only unless documented otherwise.
5. P3 remained docs-only. Roadmap wording was qualified to distinguish
   container compatibility from current CPU multi-box execution support, and no
   runtime error-text adjustment was required.

## Deferred Out-of-Scope Topics

- Full multi-box CPU condensation or coagulation implementation details.
- First-class all-box execution semantics for GPU strategy kernels.
- Container schema changes for representing environment boxes differently.
