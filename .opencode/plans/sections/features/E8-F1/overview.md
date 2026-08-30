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

## User Stories

- As a resident-simulation developer, I want an immutable capture signature so
  that changes to device, dimensions, schedule, communication, or buffer
  identities are rejected before graph launch.
- As an operator, I want explicit invalidation and recapture states so that a
  stale graph is never replayed or silently replaced.
- As a downstream E8 implementer, I want precise capability and lifecycle
  errors so that unsupported hardware is distinguishable from invalid resident
  state and post-launch failure.
