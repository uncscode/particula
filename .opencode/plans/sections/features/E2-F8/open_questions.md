# E2-F8 Open Questions

## Resolved Answers

1. CPU coagulation should immediately reject `ParticleData.n_boxes != 1` for
   unsupported paths. Avoid silent box-0 behavior.
2. Use `IsothermalCondensationStrategy.step()` plus one public pressure-delta or
   mass-transfer path that already flows through `_require_single_box` for
   representative multi-box rejection tests.
3. Include caller-managed per-box loop pseudocode only if it is accurate without
   introducing new helper APIs.
4. Mention E2-F2 environment containers as a short cross-reference in the CPU
   dynamics boundary doc. Do not imply CPU dynamics have full environment-aware
   multi-box support.

## Deferred Out-of-Scope Topics

- Full multi-box CPU condensation or coagulation implementation details.
- First-class all-box execution semantics for GPU strategy kernels.
- Container schema changes for representing environment boxes differently.
