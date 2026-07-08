# E3-F2 Documentation Updates

## Required Updates

- Update `docs/Features/Roadmap/data-oriented-gpu.md` with the final E3-F2
  decision: selected hardening design or measured accepted limitation.
- Include mixed-scale fixture details, acceptance-rate metric definition,
  stochastic tolerance approach, and focused reproduction commands.
- Cross-reference E3-F1 RNG-state semantics if repeated-step tests or diagnostic
  commands depend on persisted `rng_states`.

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
