# ADR-018: Resident Communication Integration

**Status:** Accepted
**Date:** 2026-08-09
**Decision Makers:** ADW Development Team
**Technical Story:** [#1511](https://github.com/Gorkowski/particula/issues/1511)

## Context

E7-F7 provides direct GPU primitives for gas communication, particle
communication, and prescribed volume evolution. A resident complete loop needs
to use those primitives without repeating P1 payload validation, replacing
pinned storage, or broadening the dependency-neutral scheduler or public API.
Resident checkpoints must also preserve enough closed-map state to restart a
communication-enabled session with fresh identities.

### Problem Statement

Integrate one closed-map GAS or PARTICLES communication family into the
resident lifecycle and twelve-node schedule while preserving direct-kernel
ownership, exact identity binding, and the no-transfer/no-sync resident
contract.

### Forces

**Driving Forces:**
- Communication must run before prescribed volume evolution using pre-update
  volumes.
- Normal resident steps need metadata-only validation after one acquisition.
- Restart must reconstruct, rather than reuse, pinned communication resources.

**Restraining Forces:**
- `particula.execution.scheduler` remains declaration-only and
  dependency-neutral.
- Combined maps and open endpoints are not valid resident communication forms.
- A writer failure after launch cannot guarantee rollback.

## Decision

Use concrete, direct-import-only resident composition for closed-map
communication:

1. `GPUResourceRegistry.acquire_communication()` is the sole P1
   payload-validation and allocation point. It pins exactly one complete GAS or
   PARTICLES configuration, maps, native work record, and optional final-volume
   sidecar by identity.
2. `particula.execution.resident_communication` validates only the active,
   published binding metadata and dispatches communication before optional volume
   evolution. It does not reacquire resources or inspect payloads.
3. Schema-v2 checkpoints persist either no communication family or one complete
   matching closed-map family and metadata. Restart creates fresh configuration,
   buffers, registry binding, and resident identities on the exact target
   device.
4. The resolver-produced complete loop has twelve nodes. Its first barriers are
   communication and optional volume evolution; each invalidates only
   `SATURATION_RATIO`, leaving vapor pressure fresh.

### Chosen Option

**Option 2: Pinned closed-map resident composition**

Keep native communication primitives separate and compose them only through
identity-bound registry, checkpoint, graph, and scheduler seams.

## Alternatives Considered

### Option 1: Revalidate and allocate communication resources each step

**Pros:** A simpler one-call surface for callers.

**Cons:** Repeats payload inspection and allocation, weakens resident identity
guarantees, and risks normal-step synchronization or replacement behavior.

**Reason for Rejection:** Resident execution must reuse a prevalidated, pinned
binding.

---

### Option 2: Pinned closed-map resident composition (chosen)

**Pros:** Preserves direct-kernel ownership, makes resource lifetime explicit,
and supports faithful fresh-identity restart.

**Cons:** Requires concrete request carriers and strict binding preflight.

**Reason for Selection:** It adds bounded resident integration without changing
the public execution or direct-kernel APIs.

---

### Option 3: Add communication to the public scheduler or execution exports

**Pros:** Provides a discoverable high-level API.

**Cons:** Couples dependency-neutral selection to Warp resources and implies a
broader runtime orchestration contract.

**Reason for Rejection:** The integration is intentionally concrete-only.

## Rationale

Acquisition establishes the only point that may validate map payloads or
allocate omitted work storage. Thereafter, identity and metadata validation
keeps normal resident loops bounded and avoids hidden host/device work.
Placing barriers first yields deterministic pre-process state: communication
uses the old volumes, then optional volume evolution preserves extensive
inventories. Neither operation changes vapor-pressure inputs, so only saturation
ratio becomes stale and existing consumer refresh windows remain authoritative.

### Trade-offs Accepted

1. **Closed topology only:** Combined maps and open source/sink endpoints are
   rejected for resident use.
2. **No recovery transaction:** A post-launch writer failure faults the session
   without retry or rollback.
3. **Schema complexity:** Schema-v2 records carry communication metadata and
   payloads while schema-v1 remains noncommunication-only.

## Consequences

### Positive

- GAS and PARTICLES communication work storage has an explicit pinned owner.
- The twelve-node loop has a deterministic communication-then-volume barrier.
- Checkpoint restart retains the communication contract without source-identity
  reuse.

### Negative

- Callers must use concrete imports and provide exact active bindings.
- Communication-enabled checkpoint validation has additional schema and
  nonaliasing requirements.

### Neutral

- Direct GPU communication primitives retain their launch-time validation and
  mutation authority.
- Vapor-pressure freshness behavior is unchanged; saturation refresh remains at
  existing consumers.

## Implementation

### Required Changes

1. **Registry and checkpoints**
   - Pin one complete communication family and serialize schema-v2 metadata and
     payloads.
   - Restore fresh identities through normal registry acquisition.
2. **Graph and resident executor**
   - Resolve the closed twelve-node barrier graph and dispatch exact native
     calls by mode.
   - Mark only saturation ratio stale after successful barriers.
3. **Scheduler lifecycle**
   - Preflight bindings before opening one guard token.
   - Fault the resident session if a communication or volume writer may have
     launched and fails.

### Testing Strategy

Test GAS and PARTICLES acquisition, metadata-only normal execution, checkpoint
v1/v2 compatibility and fresh restart identities, canonical barrier order,
saturation-only invalidation, no-op cases, and prelaunch versus post-launch
failure lifecycle behavior. Use Warp CPU as the baseline and skip CUDA when it
is unavailable.

### Rollback Plan

Remove the concrete resident integration and schema-v2 communication support.
No package export or direct communication-kernel API migration is required.

## Validation

### Success Criteria

- [x] A resident registry pins only one complete closed-map GAS or PARTICLES
  family and normal execution validates it without P1 revalidation.
- [x] Communication then optional volume evolution lead the twelve-node loop
  and invalidate saturation ratio only.
- [x] Schema-v2 checkpoints restart communication resources with fresh
  identities on an exactly equal device.

## References

- [ADR-007: Resident Session Checkpoint, Finalize, and Restart Boundary](ADR-007-resident-session-checkpoint-finalize-restart.md)
- [ADR-011: Resident Thermodynamic Freshness Coordinator](ADR-011-resident-thermodynamic-freshness-coordinator.md)
- [ADR-012: Resident Complete Loop and Closed Diagnostics](ADR-012-resident-complete-loop-and-diagnostics.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1511](https://github.com/Gorkowski/particula/issues/1511)

## Notes

This ADR extends ADR-007, ADR-011, and ADR-012; it supersedes none.
