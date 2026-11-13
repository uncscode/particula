# Overview

## Problem Statement

E6-F3 provides neutral spherical and rectangular wall-loss kernels, fixed-slot
removal, and persistent RNG ownership. E6-F4 extends that direct GPU path with
the charged behavior already defined by `ChargedWallLossStrategy` while
preserving subtle CPU semantics: image-charge
enhancement remains active for nonzero particle charge when wall potential is
zero, field and potential contribute signed drift, and a zero-charge particle
must exactly follow the neutral coefficient and stochastic path.

## Value Proposition

E6-F4-P1 through P6 are shipped. P1 freezes the charged-mode configuration without
changing the direct step's ownership, mutation, coefficient, or RNG contracts.
P2 adds private fp64 Warp Coulomb self-potential-ratio and image-charge
enhancement primitives, with independent NumPy/Warp parity and clipping tests.
P3 adds private fp64 Warp helpers for geometry scale, spherical/rectangular
electric-field resolution, signed mobility drift, and safely clipped charged
coefficient composition, also with independent tests.
`NeutralWallLossConfig` remains the sole concrete-module-only configuration
type. P4 connects the private helpers inside geometry-specialized charged
removal kernels: nonzero charge uses image enhancement and signed field drift,
while charged zero-charge slots retain the exact E6-F3 neutral coefficient and
RNG path. No public entry point, export, runnable, container field, transfer,
 or RNG stream was added. P5 adds regression evidence only in
 `particula/gpu/kernels/tests/wall_loss_parity_test.py`: independent charged
 CPU/Warp coefficient parity, exact zero-charge neutral fallback and ownership
 checks, invalid-call/no-mutation regressions, and exact-binomial charged
 survival validation. It makes no production API or kernel change.
P6 is the documentation-only closeout for #1414: it records the direct charged
configuration and geometry semantics, caller-owned state and mutation limits,
focused commands, and supported/deferred scope. It changes no code, API, or
test behavior.

## User Stories

- As a physics developer, I want charged inputs validated and owned
  unambiguously before future device physics is introduced.
- As a simulation developer, I want zero-charge particles to use the neutral
  E6-F3 behavior exactly so enabling charged configuration does not perturb
  neutral populations.
- As a GPU maintainer, I want validation to complete before particle or RNG
  mutation so malformed charged inputs cannot partially alter caller state.

## Parent Context

This is parent epic E6 track T4. It depends on the shipped E6-F3/T3 foundation,
is documented by E6-F4/P6, and is consumed by E6-F9's downstream direct-call
and explicit-transfer validation. Slot activation/exhaustion remains in
E6-F5/E6-F6, and backend orchestration remains deferred to Epic G.
