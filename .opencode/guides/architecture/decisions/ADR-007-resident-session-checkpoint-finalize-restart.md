# ADR-007: Resident Session Checkpoint, Finalize, and Restart Boundary

**Status:** Accepted
**Date:** 2026-07-28
**Decision Makers:** ADW Development Team
**Technical Story:** [#1488](https://github.com/Gorkowski/particula/issues/1488),
[#1525](https://github.com/Gorkowski/particula/issues/1525)

## Context

ADR-004 through ADR-006 establish an immutable resident-session carrier,
identity-pinned sidecar registry, and closed-step lifecycle gate. Resident GPU
work now needs an explicit, in-memory way to inspect, terminally retain, and
restart a complete resident binding without turning execution selection into a
scheduler, serializer, or migration layer.

### Problem Statement

Define a recovery boundary that captures both primary Warp state and acquired
sidecars exactly, preserves lifecycle ownership, and restarts only onto an
explicitly compatible device.

### Forces

**Driving Forces:**
- Checkpoints must be immutable host data, safe for detached inspection.
- Restart must recover GPU-only vapor pressure and acquired sidecars exactly.
- Finalization needs a terminal, repeatable lifecycle result.

**Restraining Forces:**
- The package-level execution export surface must remain unchanged.
- Hidden CPU fallback, synchronization policy, migration, and rollback would
  violate existing direct-GPU ownership boundaries.
- An in-progress guarded step cannot be snapshotted or finalized safely.

## Decision

Add concrete-only P5 checkpointing in `particula.execution.checkpoint` and
direct-import lifecycle methods on `ResidentSession`. No checkpoint, controller,
or restart name is exported through `particula.execution`, its adapters package,
or top-level `particula`.

`ResidentCheckpointController` is bound by exact identity to one active
`ResidentSession`, its `GPUResourceRegistry`, and its `ResidentStepGuard`.
`checkpoint()` requires the identity-pinned binding and a closed guard, then
returns a fresh immutable host snapshot while leaving the session active.

`finalize()` creates and caches that complete snapshot before atomically marking
the session `FINALIZED`. Later finalization calls return the exact cached object
without additional validation, readback, allocation, or upload.

Inspection carriers are detached and non-authoritative. In particular, CPU gas
inspection intentionally omits GPU-only vapor pressure; canonical immutable
payload bytes retain it for exact recovery. `restart_resident_session()` is a
direct-import-only, explicit same-device operation that materializes a fresh
compatible session, registry, guard, primaries, and sidecars from those
canonical payloads.

### Chosen Option

**Option 2: Identity-bound immutable host checkpoint with explicit same-device
restart**

Capture one complete immutable in-memory record under the existing guard and
registry invariants. Keep recovery explicit and reconstruct a new resident
binding only after complete host-side checkpoint validation.

## Alternatives Considered

### Option 1: Expose mutable session state or live Warp references

**Pros:** Avoids host-copy cost.

**Cons:** Is not a stable snapshot, leaks resident ownership, and cannot support
detached inspection.

**Reason for Rejection:** A checkpoint must not alias mutable live device state.

---

### Option 2: Identity-bound immutable host checkpoint with explicit same-device restart

**Pros:** Preserves exact recovery data and current ownership/lifecycle seams.

**Cons:** Requires host memory approximately equal to resident payload bytes,
plus inspection copies.

**Reason for Selection:** It provides bounded explicit recovery without changing
the execution selection or GPU transfer contracts.

---

### Option 3: Automatic migration, fallback, or persistent serialization

**Pros:** Could offer broader recovery targets.

**Cons:** Commits device-selection, transport, compatibility, and recovery policy
outside the resident-session scope.

**Reason for Rejection:** This P5 boundary is in-memory and same-device only.

## Consequences

### Positive

- A complete resident binding has immutable, detached inspection data.
- Terminal finalization is idempotent and O(1) after its first success.
- Canonical payloads recover GPU-only vapor pressure and acquired sidecars.

### Negative

- Snapshotting incurs host-copy memory and explicit synchronization cost.
- Restart creates fresh identities and requires an explicitly compatible device.

### Neutral

- No package exports, scheduler, fallback, migration, disk/remote serialization,
  broader RNG-stream policy, or execution-adapter behavior is added.
- Failures before restart setup publish no session; there is no rollback guarantee
  after a successful asynchronous device launch.

## Implementation

### Required Changes

1. **Checkpoint records and controller** (`particula/execution/checkpoint.py`)
   - Store versioned descriptors, immutable primary and sidecar payload bytes,
     detached inspection containers, metadata, and guard counters.
   - Validate checkpoint completeness before setup or device allocation.
2. **Resident lifecycle hooks** (`particula/execution/gpu_session.py`)
   - Add nonterminal checkpoint and terminal idempotent finalization delegation.
   - Permit counter restoration only on an exact fresh, closed guard binding.
3. **Registry enumeration seam** (`particula/execution/gpu_resources.py`)
   - Enumerate already-acquired bindings deterministically without copying,
      synchronization, allocation, or mutation.

4. **Schema-v3 RNG continuation** (`checkpoint.py`, `gpu_resources.py`)
   - Capture at most the canonical published coagulation and wall-loss streams
     after the checkpoint's single synchronization boundary.
   - Keep stream metadata and current little-endian `uint32` words immutable;
     current words are restart authority, while ordinary sidecar payloads exclude
     RNG roles.
   - Reconstruct fresh same-device arrays and stream bindings without reseeding.
     Normal acquisition returns those bindings by identity; explicit reset alone
     derives replacement words from the root seed.

### Testing Strategy

Verify immutable payloads, detached inspection data, canonical vapor-pressure
recovery, schema-v3 current-word continuation, identity freshness after restart,
explicit-reset-only reseeding, terminal-cache identity, closed-step rejection,
malformed-checkpoint preflight, and absence from package exports. Run the focused
checkpoint/session/resource tests and `mkdocs build --strict`.

### Rollback Plan

Remove the concrete-only checkpoint module and lifecycle hooks. No public export,
resident schema migration, fallback route, or persistent checkpoint format is
introduced.

## Validation

### Success Criteria

- [x] Checkpoint and restart data contain no live Warp or mutable NumPy aliases.
- [x] Finalization returns its terminal cached checkpoint by identity.
- [x] Restart is explicit, same-device, and recovers canonical vapor pressure.
- [x] Checkpoint APIs remain concrete-only and package exports are unchanged.
- [x] Schema-v3 continuation preserves current published-stream words without
  normal-dispatch readback or implicit reseeding.

## References

- [ADR-004: Concrete GPU-Resident Session Boundary](ADR-004-concrete-gpu-resident-session-boundary.md)
- [ADR-005: One-Time GPU Resident-Session Setup](ADR-005-one-time-gpu-resident-session-setup.md)
- [ADR-006: Resident GPU Step Lifecycle Guard](ADR-006-resident-gpu-step-lifecycle-guard.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1488](https://github.com/Gorkowski/particula/issues/1488)
- [Issue #1525](https://github.com/Gorkowski/particula/issues/1525)

## Notes

This ADR extends ADR-004 through ADR-006 and supersedes none. It intentionally
does not define a scheduler, ordinary-session automatic restart, or a general
fault-recovery policy.
