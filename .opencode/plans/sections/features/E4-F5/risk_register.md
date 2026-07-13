# Risk Register

| Risk | Impact | Mitigation | Owner |
|---|---|---|---|
| Particle per-particle units are mixed with gas kg/m3 units | Critical conservation defect | Specify concentration/volume conversion and parity-test CPU fixtures | E4-F5 implementer |
| Independent clamps make particle and gas deltas disagree | Critical mass creation/loss | Finalize once; derive every mutation and energy value from that transfer | E4-F5 implementer |
| Parallel reductions are nondeterministic | Backend parity failures | Fixed reduction order/scratch and explicit fp64 tolerance evidence | GPU maintainer |
| Disabled species enter reductions before gating | Inventory/energy contamination | Gate before all reductions; strict unchanged tests | E4-F5 implementer |
| Gas updated only after the whole call | Stale later substeps | Mutate and refresh after each of exactly four substeps | E4-F3/F5 owners |
| New buffers allocate in repeated execution | Graph/performance regression | Caller-owned fixed-shape scratch with preflight validation | GPU maintainer |
| Documentation overclaims production readiness | User-visible unsupported behavior | Keep issue #1272 gate and E4-F6/F7 dependencies explicit | Docs owner |
