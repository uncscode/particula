# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|---|---|---|---|---|
| GPU step freezes semantics before E6-F1 finishes | High: CPU/GPU divergence | Medium | Block P1 contract finalization on T1; reuse its fixtures and documented equation | E6-F2 implementer |
| Scalar and per-box normalization introduces a hidden host transfer | High: breaks residency contract | Medium | Accept Python/NumPy floating scalars or same-device Warp arrays only; reject host arrays; test helper call order | GPU API owner |
| Particle updates succeed before gas failure | High: inconsistent aerosol state | Low | Complete structural/domain/state preflight before either mutation; snapshot invalid calls | Kernel implementer |
| In-place scaling changes protected fields or array identity | High: corrupts fixed-shape state | Low | Restrict write kernels to concentration fields and assert every field/identity invariant | Test owner |
| Finite-step formula differs numerically from CPU | High: parity failure | Medium | Use one independent T1 oracle and record explicit fp64 tolerances across repeated and multi-box cases | Scientific maintainer |
| Nonuniform per-box indexing broadcasts incorrectly | High: cross-box scientific error | Medium | Include distinct coefficients and sentinel concentrations per box/species/particle | Test owner |
| CUDA availability is treated as mandatory | Medium: blocks portable CI | Low | Require Warp CPU and parameterize CUDA as a clean optional skip | CI owner |
| Scope expands into runnable/scheduler/performance work | Medium: delays Epic F | Medium | Keep direct low-level API and explicit Epic G/deferred boundaries in code review and docs | E6 owner |
