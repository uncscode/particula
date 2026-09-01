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

## Implemented P2 Boundary

Issue #1553 adds P1-bound, concrete-only setup and enqueue seams in
`particula.execution.state_updates`,
`particula.execution.thermodynamic_updates`, and
`particula.execution.diagnostics`. Setup validates the exact prepared timestep,
request/plan, graph, schedule, session, registry, primary-array, and pinning
chains, then freezes the copy or kernel inputs. Enqueue uses only those bound
records: it preserves state-writer order and empty no-ops, keeps thermodynamic
vapor/saturation freshness and cursor behavior, and dispatches diagnostics in
its fixed registration order. The standalone executor/coordinator paths remain
 available and retain their validate-then-execute contracts.

## Implemented P3 Boundary

Issue #1554 adds the concrete-only prepared resident-communication seam in
`particula.execution.resident_communication` and fixed native launch helpers in
`particula.gpu.kernels.communication`. Setup freezes one exact P1 READY
prepared timestep, its closed-map GAS or PARTICLES request and pinned resources,
duration, and optional final-volume sidecar. Enqueue dispatches communication
before a present volume barrier without validation, resource acquisition,
allocation, transfer/readback, synchronization, or dynamic map resolution.
Equal final volumes are a write-free barrier; changed volumes retain resident
concentration-scaling and volume-status behavior. Legacy executor and public
direct-kernel paths remain compatible.

## Implemented P4 Boundary

Issue #1555 splits direct GPU condensation into private setup and enqueue
boundaries. `particula.gpu.kernels.condensation` now prepares an immutable
`_PreparedCondensationCall` after preserving the public wrapper's validation,
fallback-allocation, supplied-sidecar identity, and return contracts. Its
enqueue path uses retained references only and performs the existing four equal,
gas-coupled, inventory-limited substeps without validation, allocation, host
refresh/readback, synchronization, or resource lookup.

The concrete resident adapter adds `_PreparedWarpCondensationBinding`, which
retains the prepared kernel call and its exact adapter binding. No public API,
scheduler behavior, checkpoint/resource schema, or condensation physics changed.

## User Stories

- As a resident-loop integrator, I want setup to reject incompatible state
  before capture so replay never discovers a structural error after launch.
- As a CUDA user, I want the prepared timestep to enqueue device work only so
  Warp can capture and replay the fixed sequence without hidden host work.
- As a direct-kernel caller, I want existing public entry points to remain
  validated and explicit so capture support does not weaken standalone safety.
