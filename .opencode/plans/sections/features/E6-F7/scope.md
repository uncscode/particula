# Scope

P1 delivers a CPU-only, concrete-module-only boundary that computes bounded
empirical potential event rates. It neither creates source records nor mutates
gas, particles, slots, or inventories.

## In Scope

- A `NucleationStrategy` interface and activation-type `J=A[H2SO4]` and
  kinetic-type `J=K[H2SO4]^2` strategies.
- SI-normalized inputs and outputs: precursor mass concentration in `kg/m^3`,
  molar mass in `kg/mol`, temperature in K, and `J` in `#/m^3/s`.
- Configured closed validity ranges for precursor concentration, temperature,
  and optional saturation gate; out-of-domain evaluation fails closed.
- Fixed injection composition as molecules per event (or exactly equivalent
  species mass), formation-size metadata, and optional explicit survival factor.
- Frozen validated closed intervals, validity domains, composition, and
  formation metadata; strict scalar-only evaluation and overflow-safe
  mass-to-number conversion.
- Exact zero-rate paths and optional saturation-gate semantics, with isolated
  tests for equations, boundaries, ordering, invalid inputs, immutability, and
  absent `particula.dynamics` exports.

## Out of Scope

- Full Vehkamäki binary parameterization, classical-nucleation free-energy
  solver, ion-induced/heterogeneous nucleation, cluster dynamics, chemistry, or
  automatic coefficient calibration.
- Hidden survival/growth correction; Kerminen-Kulmala correction applies only
  through a caller-supplied documented factor.
- GPU kernels (E6-F8), integrated scheduling (E6-F9/Epic G), dynamic storage,
  hidden transfers, backend selection, graph capture, differentiability, or
  performance claims.
- Silent clipping of unsupported environmental inputs, partial multi-box
  commits, or silent loss of slot-exhausted source demand.
- Source construction, inventory limitation, particle-slot admission or
  activation, gas/particle mutation, diagnostics, builders, factories,
  runnables, and all dynamics/top-level exports.
