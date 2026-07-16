# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Sequential per-mechanism selectors bias collision counts or reuse particles | Medium | Critical | Enforce one candidate loop, one summed rate, one acceptance draw, and test-local launch/RNG diagnostics | E5-F6 owner | Open |
| A component or total majorant under-bounds a pair sum | Low | Critical | Reuse proven sibling bounds, sum only non-negative bounds, enumerate every active pair in independent fixtures, and reject material ratio overflow | E5-F6 owner | Open |
| Conservative summed bound schedules excessive trials | Medium | Medium | Retain global trial cap, measure scheduled-vs-accepted counts in tests, and defer tighter bounds unless mathematically proven | E5-F6 owner | Open |
| Mixed-scale term addition loses small contributions or overflows fp64 | Medium | High | Use fp64 accumulators, mixed nanometer/droplet fixtures, finite guards, and component-vs-total diagnostics | E5-F6 owner | Open |
| Combination matrix accepts a term without its required input | Medium | High | Centralize mask-driven host preflight and snapshot all caller-owned state on failure | E5-F6 owner | Open |
| Canonical order or duplicate handling changes physical weights | Low | High | Canonicalize sets, reject duplicates, and table-test permutations | E5-F6 owner | Open |
| Charged additive merges lose charge while mass remains conserved | Low | Critical | Reuse E5-F2 charge-aware apply path and assert mass and charge separately for every charged combination | E5-F6 owner | Open |
| Additive work regresses single-term API, buffers, or RNG ownership | Medium | High | Keep one public entry point and shared launch path; run legacy, identity, reuse, and reset regressions | E5-F6 owner | Open |
| Optional CUDA behavior masks Warp CPU failure | Low | Medium | Make Warp CPU required whenever Warp is installed and treat CUDA as skip-safe additive evidence | E5-F6 owner | Open |
