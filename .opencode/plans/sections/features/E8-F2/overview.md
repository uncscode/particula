# Overview

## Problem Statement

The shipped resident scheduler validates bindings, constructs Python executors,
and invokes public direct-kernel entry points on every timestep. Those entry
points intentionally perform schema and physical-value validation and may
allocate temporary status or normalized-input arrays, synchronize through
device readback, or create selected-box arrays. This is correct for ordinary
direct calls but unsafe inside Warp graph capture. E8-F2 must divide this work
into a one-time, fail-closed setup boundary and a device-enqueue-only boundary
without weakening the existing uncaptured APIs.

## Value Proposition

An exact resident request can be prepared once, proving that its device,
shapes, process order, resource identities, controls, and physical inputs are
valid. The resulting concrete-only prepared plan then enqueues the same fixed
process sequence with no host readback, allocation, resource acquisition,
fallback, or dynamic scheduling. E8-F1 can capture that sequence, E8-F3 can pin
every required reusable buffer, and ordinary direct callers retain their
current validation and error contracts.

## Implemented P1 Boundary

Issue #1552 delivers the first setup-only seam:
`particula.execution.resident_enqueue.prepare_resident_timestep()`. It returns
an immutable, identity-semantic `PreparedResidentTimestep` only for an exact
READY E8-F1 binding after shared complete-loop metadata validation and two
signature comparisons. P1 performs neither capture nor enqueue/dispatch; later
E8-F2 phases own device-only process preparation and execution.

## User Stories

- As a resident-loop integrator, I want setup to reject incompatible state
  before capture so replay never discovers a structural error after launch.
- As a CUDA user, I want the prepared timestep to enqueue device work only so
  Warp can capture and replay the fixed sequence without hidden host work.
- As a direct-kernel caller, I want existing public entry points to remain
  validated and explicit so capture support does not weaken standalone safety.
