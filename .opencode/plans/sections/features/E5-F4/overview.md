# Overview

## Problem Statement

The direct GPU coagulation step currently executes Brownian physics only, while
the CPU codebase already defines the Seinfeld and Pandis (2016) gravitational
sedimentation kernel. Users cannot run sedimentation-driven particle-resolved
coagulation without leaving GPU-resident state, and the boundary between the
approved SP2016 model and unsupported collision-efficiency or drag variants is
not represented in the GPU API.

## Value Proposition

E5-F4 begins with a reviewable fp64 Warp implementation of mixture density,
Stokes settling with Cunningham slip correction, and the SP2016 pair kernel.
P1 is complete: the three internal helpers and independent direct Warp probes
are implemented without registering an executable sedimentation mechanism.
Later phases retain ownership of E5-F1 sampler dispatch, state safety, and
sedimentation-only execution. This gives E5-F6 a validated sedimentation term
to integrate for additive mechanisms and E5-F7 deterministic and stochastic
release fixtures.

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
