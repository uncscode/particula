# Scope

E5-F4 delivers sedimentation-only execution through the existing low-level
`coagulation_step_gpu` path. It ports only the SP2016 geometric kernel using
composition-derived particle density, Stokes settling velocity with existing
air-property/slip formulas, and collision efficiency fixed at 1.

## In Scope

- Add focused Warp helpers for per-particle effective density, settling
  velocity, and scalar SP2016 pair rate.
- Reuse the existing temperature/pressure normalization and gas-property
  helpers to derive dynamic viscosity, mean free path, Knudsen number, and
  Cunningham slip correction on the active device.
- Prove a safe sedimentation majorant, initially by exhaustively taking the
  maximum finite non-negative rate over active pairs.
- Register canonical sedimentation-only execution in E5-F1's mechanism
  capability matrix and shared bounded candidate/acceptance pass.
- Preserve fixed-shape fp64 particle data, inactive slots, caller-owned
  collision buffers, optional persistent RNG state, and the current return
  tuple.
- Add deterministic pair/property parity, bounded stochastic, multi-box,
  inactive-slot, mass-conservation, buffer-identity, RNG, and fail-before-
  mutation tests on Warp CPU; run optional CUDA cases when available.
- Document support and import limits precisely.

## Out of Scope

- Any collision efficiency other than the constant value 1, including the CPU
  placeholder `calculate_collision_efficiency_function`.
- Non-Stokes drag corrections, DNS turbulence, turbulent-shear physics, or
  sedimentation-adjusted turbulent variants.
- Brownian-plus-sedimentation and broader additive combinations; E5-F6 owns
  combination registration and one-pass total-majorant evidence.
- Binned, discrete-sectional, or continuous-PDF GPU coagulation; only the
  particle-resolved direct kernel path is supported.
- High-level `Runnable`/strategy integration, CPU fallback, dynamic slots,
  adaptive stepping, graph capture/replay, autodiff, hidden synchronization,
  or hidden CPU/device transfers.
- General parity with every CPU sedimentation strategy option, exact stochastic
  pair replay, or performance redesign.
