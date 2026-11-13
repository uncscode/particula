# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| `E2-F2` CPU `EnvironmentData` schema is missing or changes late. | E2-F3 could implement a mismatched GPU mirror. | Medium | Treat `E2-F2` as a hard dependency; confirm import path and fields before coding. | Implementer |
| Hidden transfers are accidentally introduced for convenience. | Performance regressions and unclear GPU ownership. | Medium | Keep transfers only in named conversion helpers and add documentation/tests around explicit calls. | Implementer + reviewer |
| CUDA tests become mandatory on CPU-only CI. | CI failures unrelated to feature correctness. | Low | Use existing `warp_devices(wp)` helper and skip CUDA cases when unavailable. | Implementer |
| Field dtype drift from CPU to Warp. | Round trips fail or future kernels see precision mismatches. | Low | Use `np.float64`/`wp.float64` consistently and assert dtype in tests. | Implementer |
| Existing `GasData` conversion ambiguity is copied into environment helpers. | Hard-to-review transfer semantics. | Medium | Write explicit per-field transfers and round-trip assertions for every environment field. | Reviewer |
| Updating `particula.gpu.__init__` breaks optional import behavior. | Users without Warp may see import failures. | Low | Preserve current `WARP_AVAILABLE` gating and test import behavior. | Implementer |
