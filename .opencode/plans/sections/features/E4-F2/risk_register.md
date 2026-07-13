# Risk Register

| Risk | Impact | Mitigation | Owner |
|---|---|---|---|
| “Ideal” or effective-surface mode remains ambiguous | CPU/GPU parity targets drift | Select molar ideal and one composition mode before P1/P2; encode numeric enums and fixtures | E4-F2 implementer |
| Kappa zero denominators yield NaN | Corrupt condensation state | Mirror CPU dry/zero-solute guards and test all degenerate compositions | P1 owner |
| Legacy surface callers change behavior | Backward incompatibility | Preserve `(n_species,)` static mode and regression tests | P2 owner |
| Activity or Kelvin factors applied in wrong order | Scientifically incorrect pressure delta | Independent scalar reference asserts intermediate and final values | P3 owner |
| Validation occurs after launch | Partial state mutation | Complete preflight validation and snapshot failure tests | P3 owner |
| Added per-particle loops reduce performance | GPU regression | Reuse fixed-shape buffers, avoid allocation/host transfer, leave optimization evidence to later epic tracks | E4 owner |
| CUDA unavailable in CI | Missing optional signal | Require Warp CPU and use explicit clean CUDA skips | P4 owner |
