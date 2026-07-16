# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Diameter/radius factor is ported incorrectly, producing an 8x error | Medium | High | Encode the CPU equation literally and test hand-calculated unequal-radius fixtures plus cubic scaling | P1 implementer |
| Fluid density is inferred from pressure or shared environment state instead of explicit caller input | Medium | High | Require and normalize an explicit per-box argument; add heterogeneous boxes whose density is not ideal-gas-derived | P2 implementer |
| A heuristic majorant under-bounds a valid pair | Low | Critical | Use an exhaustive maximum or prove the two-largest-radius bound; enumerate every fixture pair against it | P3 implementer |
| Invalid input mutates outputs or advances persistent RNG before failure | Medium | High | Complete all host preflight before allocation/initialization/launch and use full state snapshots | P2/P3 implementers |
| Zero dissipation semantics conflict with the positive-input contract | Medium | Medium | Resolve before implementation; default to the epic's positive-finite physical-input policy and test the chosen error | Feature owner |
| New keyword inputs break legacy Brownian callers | Low | High | Keep them keyword-only/optional and require them only for enabled turbulent shear; run legacy regression tests | P2 implementer |
| Documentation overstates ST1956 as DNS or general turbulence support | Medium | High | Use explicit no-DNS wording in API docs, support tables, examples, and review checklist | P4 docs owner |
| CUDA-only evidence hides CPU/device portability issues | Low | Medium | Require Warp CPU when installed; make CUDA additive and skippable | Test owner |
| Exhaustive majorant cost scales quadratically | Medium | Medium | Prefer the proved monotone two-largest-radius bound when reviewable; otherwise accept correctness-first O(n^2) scope and defer optimization without performance claims | P3 implementer |

Residual risk is bounded to the direct low-level particle-resolved ST1956 path.
DNS turbulence, combinations before E5-F6, and performance guarantees remain
explicitly unsupported rather than partially implemented.
