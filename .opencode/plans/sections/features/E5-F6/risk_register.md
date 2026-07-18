# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Sequential per-mechanism selectors bias collision counts or reuse particles | Medium | Critical | Shared selector/apply path and existing selector/RNG diagnostics substantiate mitigation | E5-F6 owner | Mitigated |
| A component or total majorant under-bounds a pair sum | Low | Critical | Existing independent fixtures enumerate active pairs and reject material ratio overflow | E5-F6 owner | Mitigated |
| Conservative summed bound schedules excessive trials | Medium | Medium | Retain global trial cap, measure scheduled-vs-accepted counts in tests, and defer tighter bounds unless mathematically proven | E5-F6 owner | Open |
| Mixed-scale term addition loses small contributions or overflows fp64 | Medium | High | Use fp64 accumulators, mixed nanometer/droplet fixtures, finite guards, and component-vs-total diagnostics | E5-F6 owner | Open |
| Combination matrix accepts a term without its required input | Medium | High | Existing mask-driven preflight and invalid-call state snapshots substantiate mitigation | E5-F6 owner | Mitigated |
| Canonical order or duplicate handling changes physical weights | Low | High | Existing canonicalization and permutation coverage substantiate mitigation | E5-F6 owner | Mitigated |
| Charged additive merges lose charge while mass remains conserved | Low | Critical | Existing charge-aware apply and separate mass/charge conservation coverage substantiate mitigation | E5-F6 owner | Mitigated |
| Additive work regresses single-term API, buffers, or RNG ownership | Medium | High | Existing public-path identity, reuse, reset, and legacy regressions substantiate mitigation | E5-F6 owner | Mitigated |
| Optional CUDA behavior masks Warp CPU failure | Low | Medium | Make Warp CPU required whenever Warp is installed and treat CUDA as skip-safe additive evidence | E5-F6 owner | Open |
