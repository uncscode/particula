# Overview

E8-F1 is the first feature track in parent epic E8, **Graph Capture and
Performance**. It defines the concrete contract that later E8 tracks use to
capture and replay the fixed twelve-node GPU-resident timestep without creating
a second scheduler or weakening existing ownership and failure rules.

## Problem Statement

The resident scheduler already pins state and sidecar identities, validates a
canonical process order, and executes one timestep at a time. It does not yet
have a vocabulary for graph-capture support, a captured-graph lifecycle, or a
fail-closed way to decide when structural drift requires recapture. Without
those contracts, later work could accidentally imply CPU capture support,
silently reuse a stale CUDA graph, reseed persistent RNG streams, or hide
fallback and allocation inside replay.

## Value Proposition

This feature establishes one explicit, testable source of truth for capture
capability, compatibility, lifecycle, invalidation, and recapture. E8-F2 and
E8-F3 can then separate setup work and complete resource pinning against a
stable contract, while E8-F4 through E8-F8 can validate, document, benchmark,
and profile the same lifecycle without inventing incompatible rules.

## Implementation Status

E8-F1-P1 was delivered for issue #1547. It adds the concrete-only,
Warp-import-free `particula/execution/graph_capture.py` declaration boundary
and `particula/execution/tests/graph_capture_test.py`. The delivered boundary
resolves caller-probed capability and creates/compares exact identity-based
`ResidentSimulationRequest` signatures without capture, replay, acquisition,
payload inspection, transfer, or synchronization.

E8-F1-P2 was delivered for issue #1548. The same direct-import-only module now
provides immutable host-only lifecycle metadata for `READY`, `CAPTURED`,
`INVALIDATED`, `FAULTED`, `RETIRED`, and `CLOSED` states. Explicit transition
operations preserve P1 capability/signature identities, retain the first drift
reason, classify read-only versus writer-may-have-launched failures, and permit
renewal only after retirement. They neither capture nor replay a native graph,
inspect a resident binding, nor mutate a resident session; those integration
gates remain P3 work.

## User Stories

- As a resident-simulation developer, I want an immutable capture signature so
  that changes to device, dimensions, schedule, communication, or buffer
  identities are rejected before graph launch.
- As an operator, I want explicit invalidation and recapture states so that a
  stale graph is never replayed or silently replaced.
- As a downstream E8 implementer, I want precise capability and lifecycle
  errors so that unsupported hardware is distinguishable from invalid resident
  state and post-launch failure.
