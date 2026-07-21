# Open Questions

- [ ] Which bounded initial nucleation-rate model and critical-cluster
  parameterization should become the public CPU reference?
  - Open: resolve in E6-F7 with equations, units, citations, and validity range.
- [ ] Which distribution moments must default resampling preserve beyond total
  number and per-species mass?
  - Open: resolve before E6-F6 implementation and encode in an independent CPU
    oracle.
- [ ] Should half-active particle slots be rejected universally or normalized
  by a documented shared helper?
  - Open: resolve in E6-F5 before activation APIs are frozen.
- [ ] What statistical sample sizes and tolerances are sufficient for neutral
  and charged wall-loss survival evidence?
  - Open: derive from CPU coefficients in E6-F3 and E6-F4; do not require exact
    RNG-sequence parity.
- [ ] Which exact diagnostics fields are reusable across activation,
  exhaustion, and nucleation?
  - Open: resolve in E6-F5 while retaining caller ownership and per-box shape.
