# E3-F2 Documentation Updates

## Required Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the final E3-F2
  decision: selected hardening design or measured accepted limitation.
- Keep the mixed-scale rejection-sampling outcome evidence-driven: the roadmap
  must summarize the benchmark/test artifacts that justify the selected design
  or accepted limitation instead of treating the choice as predetermined.
- Include mixed-scale fixture details, acceptance-rate metric definition,
  stochastic tolerance approach, and focused reproduction commands.
- Cross-reference E3-F1 RNG-state semantics if repeated-step tests or diagnostic
  commands depend on persisted `rng_states`.

## P1 Shipped State

- Issue #1241 did not change user-facing documentation because the shipped work
  was limited to test-local fixture/diagnostic coverage.
- No public API or production transfer-boundary language changed in this phase,
  so roadmap/documentation updates remain deferred to E3-F2-P4 unless later
  phases promote the characterization into a documented product decision.

## P2 Shipped State

- Issue #1242 updated production kernel internals in
  `particula/gpu/kernels/coagulation.py`, but the shipped change remained an
  internal selector hardening with no public API, transfer-boundary, or
  user-facing workflow change.
- The landed evidence lives in
  `particula/gpu/kernels/tests/coagulation_test.py` through selector-validity,
  sparse/degenerate, exactly-two-active fallback, accepted-count bounds, and
  mixed-scale conservation regressions.
- User-facing roadmap or feature-doc updates are still deferred to E3-F2-P4,
  when P3/P4 decide whether the final story is a measured improvement or a
  documented bounded limitation.

## Optional Updates

- Add a focused `docs/Features/` note if the selected design needs more space
  than the roadmap entry allows.
- Update `docs/Features/data-containers-and-gpu-foundations.md` only if a public
  diagnostic API is introduced; otherwise preserve existing transfer-boundary
  language.

## Documentation Acceptance

- Documentation states whether behavior was improved or explicitly bounded.
- Commands are copy/paste reproducible.
- No production path is described as performing implicit CPU/GPU transfer or
  synchronization.
