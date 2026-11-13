# Success Metrics

- [x] All seven feature tracks ship in dependency order; E4-F2/E4-F3 are the only parallel branch.
- [x] Supported vapor-pressure modes refresh on-device before each of four substeps.
- [x] Ideal and kappa activity and selected effective surface physics match CPU fixtures within documented tolerances.
- [x] Zero latent heat reproduces the isothermal path; signed energy satisfies `Q = delta_mass * L`.
- [x] Gas plus particle mass is conserved per box/species under condensation and evaporation, including inventory limits and partitioning-off cases.
- [x] Caller-owned scratch can be reused without allocation-, alias-, or stale-state regressions.
- [x] Warp CPU parity passes whenever Warp is installed; CUDA parity passes when CUDA is available and otherwise skips cleanly.
- [x] Graph-readiness checks confirm fixed launch structure and stable buffer shapes.
- [x] Every implementation track lands self-contained tests with no coverage-threshold reduction and at least 80% project coverage.
- [x] Support matrix, runnable example, troubleshooting, and focused reproduction commands accurately describe supported and unsupported behavior.
