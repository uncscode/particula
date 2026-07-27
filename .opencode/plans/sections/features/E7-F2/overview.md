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

## Shipped P1–P2 Foundation

Issue #1470 shipped E7-F2-P1's dependency-neutral metadata foundation in
`particula.execution`: immutable condensation configuration vocabularies,
exact four-axis capability requirements, and a declarative catalogue of 36 CPU
and 8 Warp-profile configurations. This is direct-module-only semantic support
metadata, not a selected execution workflow. Runtime availability, native device
handling, adapter selection, exports, and GPU APIs remain unchanged.

Issue #1471 shipped E7-F2-P2. `particula.execution` is now a package while
retaining its exact ten-name selection `__all__`. Concrete-only carriers at
`particula.execution.adapters.condensation` retain CPU or lazy Warp resources
by identity, perform ordered read-only metadata and writable-output ownership
checks, and do not select, execute, transfer, allocate, or synchronize.

Issue #1472 shipped E7-F2-P3. The same concrete-only module now supplies
selected isothermal CPU and Warp adapters: each performs exact local preflight,
makes one native backend call, and normalizes its native result while preserving
caller-owned identity. The Warp path resolves its kernel lazily and performs no
transfer, synchronization, fallback, or recovery.

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
