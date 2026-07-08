# E3-F2 Scope

## In Scope

- Add a reproducible mixed NPF/droplet GPU particle fixture spanning nanometer
  and droplet-scale masses/radii.
- Add test/debug-only acceptance-rate diagnostics for the current rejection
  sampler, including attempted and accepted collision counts where feasible.
- Prototype and evaluate a bounded hardening path, such as fixed size-bin
  majorants or stratified pair selection, inside the existing GPU coagulation
  kernel structure.
- Compare current and hardened behavior with statistical Brownian-rate checks,
  stochastic tolerance checks, and mass conservation checks.
- Document the selected design or an explicitly accepted limitation with
  reproduction commands and measured bounds.

## Out of Scope

- Adding new coagulation physics or changing the Brownian kernel formulation.
- Introducing hidden host readbacks, implicit synchronization, or automatic
  CPU/GPU data transfers in production code.
- Replacing the one-thread-per-box model with a broad parallel collision-pair
  architecture.
- Finalizing behavior that conflicts with the E3-F1 RNG-state contract.
- Creating a standalone testing-only phase; unit tests ship with the phases that
  introduce their related fixture, diagnostics, or implementation changes.

## Cross-Feature Boundaries

- Depends on E3-F1 for persisted RNG-state semantics.
- Provides measured low-level behavior for later E3 feature tracks that rely on
  stochastic GPU coagulation correctness.
