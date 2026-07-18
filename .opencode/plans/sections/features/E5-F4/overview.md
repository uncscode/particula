# Overview

## Problem Statement

The direct GPU coagulation step currently executes Brownian physics only, while
the CPU codebase already defines the Seinfeld and Pandis (2016) gravitational
sedimentation kernel. Users cannot run sedimentation-driven particle-resolved
coagulation without leaving GPU-resident state, and the boundary between the
approved SP2016 model and unsupported collision-efficiency or drag variants is
not represented in the GPU API.

## Value Proposition

E5-F4 ships the public direct-kernel configuration
`CoagulationMechanismConfig(("sedimentation_sp2016",))` for particle-resolved
fp64 Warp data. It uses the bounded active-pair scheduler, exhaustive compact
majorant, and persistent RNG/output ownership contracts. A sedimentation-
specific physical-domain preflight rejects invalid mass, concentration, or
density before output allocation/writes, RNG work, or particle mutation.
Additive sedimentation masks and other variants remain unsupported and fail at
the capability boundary.

## User Stories

- As a scientific user, I want to execute SP2016 sedimentation coagulation on
  GPU-resident particle-resolved data so that I avoid a CPU fallback or hidden
  state transfer.
- As a model reviewer, I want deterministic CPU/Warp pair evidence and an
  explicit unit-efficiency contract so that the implemented equation and its
  limits are auditable.
- As an integrator, I want unsupported distributions, efficiencies, and model
  variants rejected before particle, output-buffer, or RNG mutation so that a
  failed call is state-safe.

Parent epic: E5, GPU Coagulation Physics Coverage. Track: T4. Classifier
diagnostics: none.
