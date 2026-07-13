# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|---|---|---|---|---|
| Tests mirror GPU equations and miss shared defects | High | Medium | Assemble multi-box expectations from independent one-box CPU equations | P1 implementer |
| Aggregate conservation hides box/species leakage | High | Medium | Assert invariants separately per box and species | P2 implementer |
| Strict tolerance is scale-inappropriate | Medium | Medium | Use explicit fp64 fixtures, absolute floors, and documented scale analysis; never relax parity to mask bookkeeping defects | P2 implementer |
| CUDA-only behavior escapes CI | High | Medium | Keep Warp CPU mandatory, share device matrix, and document optional CUDA command | P1 implementer |
| Capture accidentally includes allocation or transfer | High | Medium | Preallocate complete scratch and inputs; assert identity and capture/replay equivalence | P3 implementer |
| Autodiff evidence is overstated | High | Medium | Bound tests to smooth interiors and publish clamp/in-place non-claims | P4 implementer |
| Upstream E4 APIs change after fixtures are written | Medium | Medium | Gate final fixture/tolerance design on E4-F1 through E4-F5 completion | Feature owner |
| Optional backend skips conceal mandatory CPU failure | High | Low | Assert CPU device presence whenever Warp is installed | P1 implementer |
