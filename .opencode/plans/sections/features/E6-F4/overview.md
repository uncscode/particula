# Overview

## Problem Statement

E6-F3 provides neutral spherical and rectangular wall-loss kernels, fixed-slot
removal, and persistent RNG ownership, but the direct GPU path cannot yet model
the charged behavior already defined by `ChargedWallLossStrategy`. A separate
charged implementation risks changing subtle CPU semantics: image-charge
enhancement remains active for nonzero particle charge when wall potential is
zero, field and potential contribute signed drift, and a zero-charge particle
must exactly follow the neutral coefficient and stochastic path.

## Value Proposition

E6-F4-P1 and P2 are shipped. P1 freezes the charged-mode configuration without
changing the direct step's ownership, mutation, coefficient, or RNG contracts.
P2 adds private fp64 Warp Coulomb self-potential-ratio and image-charge
enhancement primitives, with independent NumPy/Warp parity and clipping tests.
`NeutralWallLossConfig` remains the sole concrete-module-only configuration
type. The primitives are not public exports and are not composed into the
direct wall-loss kernel; charged execution therefore remains the E6-F3 neutral
coefficient and RNG path. Electric-field drift and charged composition remain
deferred.

## User Stories

- As a physics developer, I want charged inputs validated and owned
  unambiguously before future device physics is introduced.
- As a simulation developer, I want zero-charge particles to use the neutral
  E6-F3 behavior exactly so enabling charged configuration does not perturb
  neutral populations.
- As a GPU maintainer, I want validation to complete before particle or RNG
  mutation so malformed charged inputs cannot partially alter caller state.

## Parent Context

This is parent epic E6 track T4. It depends on E6-F3/T3 and is consumed by the
E6-F9 integrated direct-step validation. Slot activation/exhaustion remains in
E6-F5/E6-F6, and backend orchestration remains deferred to Epic G.
