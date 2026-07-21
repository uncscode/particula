# Risk Register

| Risk | Likelihood | Impact | Mitigation / Exit Evidence | Owner |
|------|------------|--------|----------------------------|-------|
| Warp transport or special-function port diverges from NumPy at small/large arguments | Medium | High | Reuse existing device primitives, add stable analytic limits, and require per-primitive plus end-to-end coefficient parity over representative particle scales | P1/P2 implementer |
| Rectangular coth or spherical Debye evaluation produces NaN/overflow near zero settling | Medium | High | Implement explicit limiting behavior and test diffusion-dominated, gravity-dominated, and finite-domain boundaries against the CPU oracle | P1 implementer |
| Stochastic tests are flaky or falsely imply exact RNG parity | Medium | High | Predeclare binomial confidence/sigma bounds, use sufficient samples and multiple seeds, separate deterministic coefficient tests, and prohibit CPU/GPU draw-order assertions | P5/P6 test owner |
| RNG is silently reset each timestep or advanced during invalid calls | Medium | High | Mirror coagulation's caller-owned `wp.uint32` lifecycle; test initialize-once reuse, explicit reset, identity, and preflight snapshots | P5 implementer |
| Removal leaves mass or charge in concentration-zero slots | Medium | Critical | Use one reviewed fixed-slot clear pass and assert every species mass, concentration, and charge is exactly zero for every loss | P4 implementer |
| Inconsistent active predicates sample half-active or sentinel slots | Medium | High | Freeze the active predicate in P3, reject invalid half-active state where required, and cover sparse/inactive fixtures without activation or compaction | P3/P4 implementer |
| Validation launches or allocates before detecting malformed caller state | Medium | High | Define deterministic preflight ordering, device scans before RNG/mutation, and snapshot all caller-owned state for parameterized failures | P3 implementer |
| Geometry API accidentally admits charged terms or ambiguous radius/dimensions | Low | High | Immutable neutral-only configuration with exact geometry variants and fail-closed exclusivity tests; charged extension remains E6-F4 | P3 reviewer |
| Scope expands into high-level GPU scheduling, transfers, or performance work | Medium | Medium | Enforce E6/Epic G boundary in API review and docs; export only the low-level direct step and make no benchmark claim | Feature owner |
| Optional CUDA behavior becomes a delivery blocker | Low | Medium | Require Warp CPU, parameterize optional CUDA through shared availability helpers, and assert stable clean skips | P6 test owner |

Risks close only when the cited tests or documentation evidence ships; prose-only
assurances are insufficient for physics, RNG, and slot-integrity risks.
