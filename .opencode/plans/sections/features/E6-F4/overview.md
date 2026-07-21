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

E6-F4 extends the E6-F3 direct, particle-resolved Warp step without changing
its ownership or mutation contracts. Users gain deterministic fp64 coefficient
parity with the CPU charged model, statistically validated survival behavior,
and an exact neutral fallback while retaining fixed shapes, explicit transfers,
caller-owned persistent RNG, and optional CUDA evidence.

## User Stories

- As a physics developer, I want charged GPU wall loss to preserve the CPU
  image-charge and electric-field equations so CPU and Warp studies have one
  behavioral reference.
- As a simulation developer, I want zero-charge particles to use the neutral
  E6-F3 behavior exactly so enabling charged configuration does not perturb
  neutral populations.
- As a GPU maintainer, I want validation to complete before particle or RNG
  mutation so malformed charged inputs cannot partially alter caller state.

## Parent Context

This is parent epic E6 track T4. It depends on E6-F3/T3 and is consumed by the
E6-F9 integrated direct-step validation. Slot activation/exhaustion remains in
E6-F5/E6-F6, and backend orchestration remains deferred to Epic G.
