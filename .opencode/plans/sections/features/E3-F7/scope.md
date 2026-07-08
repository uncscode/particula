# Scope

## In Scope

- Add a CPU-only integration test for `CondensationLatentHeat` through the public
  `MassCondensation.execute()` path.
- Adapt the existing particle-resolved condensation integration fixture style so
  the scenario remains deterministic and fast.
- Use a constant latent-heat strategy, such as `par.gas.ConstantLatentHeat`, so
  energy bookkeeping has an unambiguous reference value.
- Assert particle water mass increase, gas water decrease, total water inventory
  conservation, finite positive latent-heat release, and latent-heat energy equal
  to transferred mass times latent heat.
- Document the test as the Epic D CPU reference baseline in roadmap or feature
  documentation.
- Keep all validation CPU-backed and compatible with default integration-test
  execution.

## Out of Scope

- Implementing GPU latent-heat kernels or GPU production parity.
- Changing CPU or GPU container schemas.
- Introducing slow stochastic or performance benchmarks.
- Broad refactors of condensation strategy internals beyond any minimal helper
  extraction required for a clear test.
- Lowering coverage thresholds, relaxing existing conservation assertions, or
  marking the new baseline as optional/slow.

## Constraints

- CPU baseline only; no GPU latent-heat production parity claim.
- Default integration test must remain stable and reasonably fast.
- Co-located tests ship with each phase that changes executable behavior.
- Preserve public API usage where practical (`par.dynamics`, `par.gas`) so the
  baseline reflects user-visible behavior.
