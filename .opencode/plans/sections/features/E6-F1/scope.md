# Scope

Deliver a validated CPU dilution strategy and `RunnableABC` implementation
that update particle and gas concentrations in an `Aerosol`, retain the
existing free-function API, and establish the process-level reference required
by E6-F2.

## In Scope

- Freeze units and semantics for chamber volume [m³], inlet flow [m³/s],
  dilution coefficient [s⁻¹], time [s], particle number concentration [1/m³],
  and gas mass concentration [kg/m³].
- Validate finite positive volume, finite nonnegative flow/coefficient and
  concentration, finite nonnegative time, and positive integer `sub_steps` at
  public boundaries.
- Preserve scalar/NumPy behavior of the existing dilution helper functions.
- Add container-level CPU behavior for `ParticleRepresentation` and
  `GasSpecies`, including single- and multi-species gas state.
- Add a dilution strategy plus a composable `Dilution` runnable with explicit
  substep behavior and in-place `Aerosol` updates.
- Guarantee exact no-ops for zero flow/coefficient or zero elapsed time.
- Preserve particle mass, charge, density, distribution coordinates and
  representation volume, and preserve gas identity/metadata and atmospheric
  state.
- Export the public CPU API through `particula.dynamics` and add fast,
  co-located tests, reference documentation, and an example.

## Out of Scope

- Direct Warp kernels, CPU/Warp parity, CUDA evidence, device sidecars, or GPU
  container changes; these belong to E6-F2.
- Wall loss, slot activation/exhaustion, nucleation, and integrated process
  sequencing; these belong to E6-F3 through E6-F9.
- Inlet aerosol or gas source composition, recirculation, leaks, pressure-flow
  coupling, multi-box transport, or CFD coupling.
- Backend selection, high-level GPU runnables, process scheduling, graph
  capture, differentiability, and performance claims; these remain Epic G or
  later work.
- Scaling per-particle mass to represent dilution.
