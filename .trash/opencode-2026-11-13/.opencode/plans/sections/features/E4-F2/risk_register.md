# Risk Register

| Risk | Impact | Mitigation | Owner |
|---|---|---|---|
| “Ideal” or effective-surface mode remains ambiguous | CPU/GPU parity targets drift | Select molar ideal and one composition mode before P1/P2; encode numeric enums and fixtures | E4-F2 implementer |
| Kappa zero denominators yield NaN | Corrupt condensation state | Mirror CPU dry/zero-solute guards and test all degenerate compositions | P1 owner |
| Legacy surface callers change behavior | Backward incompatibility | Preserve `(n_species,)` static mode and regression tests | P2 owner |
| Activity or Kelvin factors applied in wrong order | Scientifically incorrect pressure delta | Resolved in P3 with independent coupled references for all supported mode pairs | P3 owner |
| Validation occurs after launch | Partial state mutation | Resolved in P3 with aggregate preflight and monkeypatched no-launch snapshots | P3 owner |
| Weighted surface reduction repeats per condensing species | GPU regression | Resolved in P3 with one step-owned reduction result per active particle | P3 owner |
| CUDA unavailable in CI | Missing optional signal | Require Warp CPU and use explicit clean CUDA skips | P4 owner |
