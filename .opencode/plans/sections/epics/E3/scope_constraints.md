# Scope and Constraints

## In Scope

- Coagulation RNG state persistence and tests around repeated low-level GPU
  kernel calls.
- Mixed NPF/droplet coagulation rejection-sampling characterization, hardening,
  or explicit limitation documentation.
- A decision record for the one-thread-per-box coagulation design, including
  measured single-box and multi-box scaling limits where practical.
- Low-level condensation and coagulation kernel import-path resolution and a
  direct GPU quick-start.
- Device-aware pytest policy for Warp CPU and CUDA-if-available validation,
  including parity and stochastic tolerance expectations.
- Runnable `CondensationLatentHeat` documentation example using existing CPU
  public APIs.
- CPU integration-level latent-heat conservation baseline for future GPU parity
  work.

## Out of Scope

- High-level backend selection, automatic backend fallback, or runnable-level
  GPU/CPU dispatch policy.
- New GPU physics, new coagulation models, or GPU latent-heat implementation.
- Hidden CPU/GPU transfers, hidden synchronization, or convenience APIs that
  obscure device ownership.
- Mandatory CUDA infrastructure; CUDA tests remain optional and conditional.
- Broad performance optimization beyond recording current design limits and
  scoping follow-up work.
- Reworking existing CPU condensation strategy architecture beyond the minimal
  docs/example/baseline needs.

## Constraints

- Follow Particula naming, typing, and Google-style docstring conventions.
- Keep tests colocated in module-level `tests/` directories using the
  `*_test.py` suffix.
- Prefer deterministic fixtures for baselines; use statistical assertions for
  stochastic coagulation behavior.
- Preserve existing explicit transfer helpers and environment validation
  contracts.
- Keep documentation user-facing and explicit about optional Warp/CUDA setup.
