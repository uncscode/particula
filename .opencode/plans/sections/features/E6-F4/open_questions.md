# Open Questions

- [x] Should nonzero charge retain image-charge enhancement when wall potential
  is zero?
  - Resolved 2026-07-21: Yes. This is an explicit CPU semantic and acceptance
    criterion; wall potential does not gate the image-charge factor.
- [x] Should configured field or potential alter a zero-charge particle?
  - Resolved 2026-07-21: No. Drift is charge-dependent and the slot must use the
    E6-F3 neutral coefficient and survival probability.
- [x] Does E6-F4 introduce a separate charged step API or RNG stream?
  - Resolved 2026-07-21: No. Charged mode extends E6-F3 configuration and reuses
    its direct step, active predicate, clearing, and persistent RNG sidecar.
- [ ] Must rectangular GPU configuration accept both scalar field magnitude and
  a three-component vector, or only the CPU-documented vector form?
  - Open: Freeze the narrowest CPU-compatible form in P1; any accepted scalar
    alias must normalize deterministically and receive explicit parity tests.
- [ ] Is zero-charge fallback required to be bitwise identical to E6-F3 or only
  numerically identical at zero tolerance after device compilation?
  - Open: P1/P5 should record the strongest stable assertion supported by the
    shared neutral branch, never a weaker generic charged tolerance by default.
- [ ] Which exact CPU/Warp `rtol`/`atol`, sample counts, and confidence bound
  cover the charged matrix without masking nanometer-scale discrepancies?
  - Open: Record them before implementation results are evaluated and include
    scale-stratified cases.
