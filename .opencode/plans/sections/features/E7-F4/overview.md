# Overview

## Problem Statement

Issue #1451 Track T4 requires a production session boundary above the shipped
direct Warp process APIs. Today callers must upload particle, gas, and
environment containers independently, allocate heterogeneous process sidecars,
retain CPU-only metadata, manage synchronization, and decide when restoration
is safe. That works for direct examples but does not provide a reusable,
validated multi-timestep abstraction. Ad hoc loops can introduce repeated
transfers, unstable allocations, stale ownership, or checkpoints that cannot be
restarted faithfully.

## Value Proposition

E7-F4 will provide typed GPU-resident session state that performs one explicit
setup upload, retains fixed-shape Warp containers and reusable sidecars across
steps, and transfers state only through explicit checkpoint or finalization
operations. Stable identities, lifecycle guards, metadata preservation, and
clear failure semantics give E7-F5 a safe scheduling substrate while preserving
all existing direct-kernel physics and ownership contracts.

## User Stories

- As a simulation user, I want particle, gas, and environment state to remain
  on my selected Warp device across timesteps so that normal stepping performs
  no bulk CPU round trips.
- As a process integrator, I want one validated registry of fixed-shape
  sidecars so that buffers and RNG resources can be reused without accidental
  reallocation or cross-device mixing.
- As an operator, I want explicit checkpoints and finalization so that I can
  inspect or restart a run at a known synchronized boundary while retaining
  CPU-only metadata such as ordered gas names.

This epic-linked feature belongs to E7 and depends on E7-F1 and E7-F6. It
implements only issue #1451 Track T4; deterministic process scheduling remains
E7-F5, transport remains E7-F7, and full RNG stream policy remains E7-F8.

## Implementation Status

P1 was implemented in issue #1484 (source plan #1460). The concrete-only
`particula/execution/gpu_session.py` boundary now retains caller-owned Warp
particle, gas, and environment containers by identity with immutable resident
dimensions, Warp device/gas-name metadata, and declared lifecycle values.
Construction is metadata-only and lazy-imports Warp; sidecars, scheduling,
lifecycle operations, checkpointing, restoration, and finalization remain future
phases.

P2 was implemented in issue #1485. The concrete-only
`setup_resident_session()` factory performs exact local `Device`/Warp-backend
and CPU cross-container schema preflight before importing conversion helpers.
It converts particles, gas, and environment once, in that order, retains ordered
CPU gas names in `ResidentMetadata`, and publishes one validated `ACTIVE`
session. It adds no exports, fallback, synchronization, restore, or sidecars.
Native-device availability remains an explicit E7-F6 upstream precondition.

P4 was implemented in issue #1487 (commit `61f101de1`). Concrete-only
`ResidentStepGuard` and frozen identity-only `ResidentStepToken` now provide
one-open-step bookkeeping for one exact active `ResidentSession` and
`GPUResourceRegistry` binding. `GPUResourceRegistry.validate_pinned_session()`
validates that exact pinned session through the existing active signature/schema
checks without resource acquisition. Matching completion alone advances the
guard's completed-step count and simulated time; the guard neither dispatches
adapters nor transfers, synchronizes, allocates, resizes, restores, or falls
back. Its `assert_step_closed()` gate is the required preflight for future P5/P6
lifecycle boundaries, while P5/P6 retain checkpoint/finalize/close/fault policy.

P5 was completed in issue #1488. Direct-import-only
`particula.execution.checkpoint` now supplies immutable, versioned in-memory
checkpoints and same-device restart. `ResidentSession.checkpoint()` is a
nonterminal, synchronized snapshot; `finalize()` caches its first checkpoint,
transitions to `FINALIZED`, and returns that exact cached record thereafter.
Canonical payload bytes preserve all 12 primary arrays (including GPU-only
vapor pressure) and every acquired sidecar, while detached CPU inspection
carriers remain non-authoritative. Checkpoint/restart remains concrete-only:
there is no package export, device selection or migration, serialization, CPU
fallback, scheduler, or RNG-policy expansion.

P6 was completed in issue #1489. The concrete-only resident-session boundary
now explicitly classifies direct-owner failures as read-only or writer-may-have-
launched. Read-only cleanup releases the exact open token without advancing
guard counters or simulated time and leaves the session `ACTIVE`; an uncertain
writer failure preserves observable device mutation, faults the exact session,
and preserves the original exception. `close()`/`discard()` provide terminal
no-restore disposal: active and faulted sessions transition to `CLOSED`, while
closed and finalized sessions are write-free idempotent cases. No package export
or adapter boundary changed.
