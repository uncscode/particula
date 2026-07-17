# Overview

## Problem Statement

The direct GPU coagulation step currently executes Brownian physics only, while
the CPU codebase already defines the Seinfeld and Pandis (2016) gravitational
sedimentation kernel. Users cannot run sedimentation-driven particle-resolved
coagulation without leaving GPU-resident state, and the boundary between the
approved SP2016 model and unsupported collision-efficiency or drag variants is
not represented in the GPU API.

## Value Proposition

E5-F4 now has the P1 helpers and a private P2 execution slice. The direct
kernel can execute only the exact sedimentation-only mask through the existing
bounded active-pair scheduler and RNG path. It uses an exhaustive compact
active-pair majorant and private, cleared settling-velocity scratch. Public
capability validation continues to reject sedimentation before allocation or
mutation, and mixed sedimentation masks remain no-ops in private dispatch.
This preserves a validated term for later additive-mechanism and public-support
work without expanding the public API.

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
