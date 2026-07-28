# Overview

## Problem Statement

Issue #1451 Track T3 requires Brownian coagulation to run through the typed
backend-selection boundary established by E7-F1 and constrained by E7-F6.
Today users must choose between the CPU `Coagulation` runnable and the direct
`coagulation_step_gpu` API themselves, including managing output buffers,
device state, validation, and per-box RNG lifetime. Reinitializing RNG state at
each timestep breaks the intended persistent stochastic stream, while hiding
that state in an adapter would make replay and checkpoint work ambiguous.

## Value Proposition

E7-F3 provides one explicit Brownian process selection contract that delegates
path preserves particle identity, exposes collision diagnostics, reuses a
caller-owned per-box RNG buffer across calls, and performs no hidden transfer,
synchronization, reseeding, or fallback. This gives E7-F5 a schedulable
coagulation node and gives E7-F8 a precise RNG seam for stream and restart
semantics.

E7-F3-P2 established the concrete carrier seam, and E7-F3-P3 now adds the
concrete-only dispatch boundary at
`particula.execution.adapters.coagulation`. Frozen CPU and resident-Warp P3
states retain the P2 resources by identity. The CPU adapter invokes
`Coagulation.execute()` once with the original controls; the Warp adapter
lazily resolves and invokes `coagulation_step_gpu()` once with the retained
resident resources and RNG intent. Both return `ExecutionResult` with
`MutationScope.STATE` and the existing typed P2 result carrier. Capability
selection, conversion, synchronization, fallback, public exports, and
user-facing documentation remain outside this completed phase.

E7-F3-P4 closes the adapter failure boundary. State construction accepts only
the fieldless, exact-type `BrownianCoagulationConfig` marker, rejecting all
other request-shaped objects before Warp import or lazy kernel resolution. Warp
P2 additionally owns finite, nonnegative selected-time validation while
retaining native ownership of particle, environment/volume, output, device,
dtype, capacity, and detailed RNG schemas. CPU validates selected execution
controls before its one runnable call; Warp resolves and calls the native entry
point once and propagates native and post-launch failures without recovery.

## User Stories

- As a simulation user, I want to request Brownian coagulation by backend so
  that I do not directly orchestrate heterogeneous CPU and Warp entry points.
- As a reproducibility-focused user, I want one explicitly seeded RNG buffer to
  persist across timesteps so that repeated runs can replay the same Warp
  stochastic sequence.
- As a maintainer, I want unsupported modes rejected before mutation so that
  backend selection never silently changes physics, transfers state, or falls
  back after a failure.
