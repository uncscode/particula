# Overview

## Problem Statement

Particula ships mature CPU condensation strategies and a bounded direct-Warp
condensation step, but users must currently choose and orchestrate those paths
themselves. Their configuration, state, return, mutation, sidecar, and failure
semantics differ enough that ad hoc selection risks hidden transfers, stale
thermodynamics, unsupported physics, or incorrect assumptions about rollback.

## Value Proposition

E7-F2 implements issue #1451 Track T2 by making one supported condensation
workflow selectable through the E7-F1 execution context after E7-F6 freezes
availability and fallback policy. It preserves the CPU implementation as the
independent reference, delegates GPU physics to `condensation_step_gpu`, and
states exactly which particle, gas, transfer, and energy objects mutate or are
returned. Isothermal and latent-heat direct variants are supported; staggered
and unsupported BAT configurations fail before mutation rather than moving data
or silently changing backend.

## User Stories

- As a simulation user, I want to request CPU or Warp condensation through one
  typed selection boundary so that backend choice does not require direct kernel
  orchestration.
- As a scientific user, I want recorded CPU/Warp tolerances and conservation
  evidence so that selected execution has a trustworthy reference.
- As a maintainer, I want explicit capability, ownership, validation, and
  unsupported-mode contracts so later resident scheduling can compose
  condensation without hidden transfers or physics rewrites.

Parent epic: [E7](../../epics/E7/vision_problem.md). Scope authority: issue
#1451 Track T2 and Epic G in `docs/Features/Roadmap/data-oriented-gpu.md`.
