# ADR-005: One-Time GPU Resident-Session Setup

**Status:** Accepted
**Date:** 2026-07-29
**Decision Makers:** ADW Development Team
**Technical Story:** [#1485](https://github.com/Gorkowski/particula/issues/1485)

## Context

ADR-004 established a direct-import-only P1 carrier for already-resident Warp
state. Process adapters also need one narrow, explicit boundary that converts
compatible CPU particle, gas, and environment carriers into one complete active
resident session without turning sessions into a public API or lifecycle manager.

### Problem Statement

Provide one deterministic CPU-to-Warp setup operation while preserving explicit
transfer ownership, CPU gas-name metadata, optional-Warp import isolation, and
the upstream execution-availability policy.

### Forces

**Driving Forces:**
- Resident adapters need complete same-device Warp state and dimensions.
- CPU gas names must remain ordered CPU metadata because `WarpGasData` has no
  name field.
- Conversion mechanics already belong to `particula.gpu.conversion`.

**Restraining Forces:**
- Repeated or hidden uploads obscure ownership and may publish partial state.
- Availability probing, fallback, and device selection belong to E7-F6.
- Public exports or lifecycle operations would commit unsupported API semantics.

## Decision

Add direct-import-only `setup_resident_session` in
`particula.execution.gpu_session` as P2 of the resident-session boundary. It
performs local CPU-only preflight, exactly one ordered upload of particle, gas,
and environment data, and publishes only a complete `ResidentSession` in
`ResidentLifecycle.ACTIVE`.

### Chosen Option

**Option 2: Execution-layer orchestration over existing conversions**

The factory will:

1. Require an exact Warp `Device`, valid CPU carrier types, matching declared
   shapes, and ordered string gas names before importing conversion helpers.
2. Rely on E7-F6's native-availability precondition rather than probing,
   normalizing, selecting, or falling back from `device.native`.
3. Call `to_warp_particle_data`, `to_warp_gas_data`, and
   `to_warp_environment_data` exactly once, in that order, using the unchanged
   native identifier.
4. Retain `tuple(gas.name)` solely in `ResidentMetadata` and construct one final
   `ACTIVE` session; conversion and final-validation errors propagate without
   publishing a partial session.

## Alternatives Considered

### Option 1: Require callers to upload and construct sessions separately

**Pros:** No new orchestration code.

**Cons:** Leaves ordering, metadata capture, and complete-session publication
inconsistent across future adapters.

**Reason for Rejection:** A single internal setup seam makes the transfer
contract explicit without changing public ownership.

---

### Option 2: Execution-layer orchestration over existing conversions (chosen)

**Pros:** Reuses the established transfer owner, retains CPU metadata correctly,
and creates a complete validated session once.

**Cons:** Adds a concrete internal factory that future callers must import
directly.

**Reason for Selection:** It is the smallest boundary that supplies resident
state while preserving P1 carrier validation and package export policy.

---

### Option 3: Add uploads, availability handling, and lifecycle management to a
public session API

**Pros:** Could centralize future operational concerns.

**Cons:** Mixes transfer, E7-F6 availability/fallback policy, synchronization,
restoration, and lifecycle semantics into an unsupported API.

**Reason for Rejection:** These concerns have separate ownership and remain
deferred.

## Rationale

`particula.gpu.conversion` remains the sole transfer-mechanics boundary, while
`ResidentSession` remains the final generated-schema/shared-device validator.
The factory supplies only ordering and all-or-nothing publication. Local CPU
preflight prevents malformed carriers from loading conversion helpers; it does
not duplicate E7-F6 native availability policy.

### Trade-offs Accepted

1. **One upload only:** Private allocations made by a failing conversion are not
   rolled back, but no partial session is returned.
2. **No availability probe:** A selected native Warp device must already be
   availability-approved by E7-F6.
3. **No operational behavior:** Synchronization, restoration, sidecars,
   lifecycle transitions, retries, cleanup, and fallback remain deferred.

## Consequences

### Positive

- Resident callers receive identity-preserving converted containers in one
  complete `ACTIVE` session.
- Gas-name ordering remains correct without extending Warp schemas.
- CPU-only malformed input rejects before GPU conversion import or mutation.

### Negative

- The factory is intentionally direct-import-only rather than a supported API.
- Availability integration depends on the upstream E7-F6 boundary.

### Neutral

- Existing direct kernels, conversion helper ownership, package exports, and P1
  lifecycle vocabulary remain unchanged.

## Implementation

### Required Changes

1. **Session factory** (`particula/execution/gpu_session.py`)
   - Preflight local CPU carriers and schemas before function-local conversion
     imports.
   - Upload each carrier once in fixed order and publish a final active session.
2. **Focused tests** (`particula/execution/tests/gpu_session_test.py`)
   - Verify import isolation, preflight rejection, one-call order, error
     propagation, input identity, and no partial publication.
3. **Architecture documentation**
   - Describe the P2 ownership and deferred-operation boundaries.

### Testing Strategy

Use CPU-only subprocess preflight tests that block Warp/GPU imports and
Warp-optional success fixtures that validate conversion call order, identity,
metadata, and final session schema validation. When E7-F6 is available, test its
unavailable-device rejection seam before conversion.

### Rollback Plan

Remove the direct-only factory and its focused tests. No public export,
persistent resource, lifecycle transition, or data migration is introduced.

## Validation

### Success Criteria

- [x] P2 performs one ordered particle/gas/environment upload after local
  CPU-only preflight.
- [x] Only a complete `ACTIVE` session is published, with CPU gas names retained
  as ordered metadata.
- [x] The factory remains unexported and introduces no fallback, synchronization,
  restoration, sidecars, or lifecycle operations.
- [x] Native availability remains an upstream E7-F6 precondition.

## References

- [ADR-003: Dependency-Neutral Execution Capability Vocabulary](ADR-003-dependency-neutral-execution-capabilities.md)
- [ADR-004: Concrete GPU-Resident Session Boundary](ADR-004-concrete-gpu-resident-session-boundary.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1485](https://github.com/Gorkowski/particula/issues/1485)

## Notes

This ADR extends ADR-004's resident-session boundary and supersedes none. Its
completed active session is the P4 guard input defined by
[ADR-006: Resident GPU Step Lifecycle Guard](ADR-006-resident-gpu-step-lifecycle-guard.md).
