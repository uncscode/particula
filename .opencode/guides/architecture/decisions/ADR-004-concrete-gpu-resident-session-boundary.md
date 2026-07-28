# ADR-004: Concrete GPU-Resident Session Boundary

**Status:** Accepted
**Date:** 2026-07-28
**Decision Makers:** ADW Development Team
**Technical Story:** [#1484](https://github.com/Gorkowski/particula/issues/1484)

## Context

GPU process adapters need a common way to retain already-resident caller-owned
Warp particle, gas, and environment containers without introducing a scheduler,
transfer layer, or public session API. The initial boundary establishes only the
invariant needed by later phases: compatible, same-device resident state is
retained by identity.

### Problem Statement

Provide an optional-Warp construction boundary that is constant-cost with
respect to `(n_boxes, n_particles, n_species)` payload size and does not alter
established execution, GPU, or package export contracts.

### Forces

**Driving Forces:**
- Future resident GPU adapters need stable dimensions, device metadata, CPU gas
  names, and lifecycle vocabulary associated with the same caller-owned state.
- Construction must reject incompatible schemas and devices before later
  operational layers use the state.
- CPU-only imports of `particula.execution` must stay independent of Warp.

**Restraining Forces:**
- Payload scans, host readbacks, synchronization, or validation kernels would
  make construction size-dependent and introduce hidden runtime effects.
- A session abstraction could prematurely imply resource ownership, scheduling,
  fallback, migration, or lifecycle operations.
- Package exports would create unsupported public API commitments.

## Decision

Add `particula.execution.gpu_session` as an intentionally concrete-only P1
boundary. It defines immutable dimensions, metadata, lifecycle vocabulary, and
`ResidentSession`; no name is re-exported through `particula.execution`, its
adapters package, or top-level `particula`.

### Chosen Option

**Option 2: Read-only concrete resident-session carrier**

Construction retains supplied Warp particle, gas, and environment containers;
dimensions; metadata; lifecycle value; and gas-name tuple by identity. It lazily
imports Warp and generated container types only after CPU-only carrier checks,
then performs fixed-cost type, dtype, shape, and device metadata validation. It
does not access payloads, convert, allocate, synchronize, launch kernels,
schedule, fall back, migrate, transition, finalize, or close.

## Alternatives Considered

### Option 1: Add session management to `particula.gpu`

**Pros:** Places resident state beside Warp schemas and conversion helpers.

**Cons:** Blurs the schema/transfer boundary and implies a GPU-package public API.

**Reason for Rejection:** `particula.gpu` retains explicit conversion and direct
kernel responsibilities; this carrier is an execution-layer precondition.

---

### Option 2: Read-only concrete resident-session carrier (chosen)

**Pros:** Establishes identity and schema invariants without operational
semantics or public export commitments.

**Cons:** Later layers must explicitly add lifecycle and execution behavior.

**Reason for Selection:** It creates the smallest stable foundation for future
resident execution while preserving explicit caller ownership.

---

### Option 3: Implement a runnable resident session now

**Pros:** Could immediately centralize dispatch and resource lifecycle.

**Cons:** Couples P1 validation to unresolved scheduling, synchronization,
failure, transition, and fallback semantics.

**Reason for Rejection:** Those concerns belong to later phases and require
separate decisions.

## Rationale

The carrier treats GPU-resident data as caller-owned opaque storage and inspects
only fixed schema metadata. Lazy optional imports preserve the dependency-neutral
package seam, while immutable fields prevent rebinding without claiming retained
Warp containers are immutable.

### Trade-offs Accepted

1. **Concrete-only import:** There is no supported public session API.
2. **Metadata-only validation:** Construction does not prove payload physics;
   direct kernels and adapters retain that duty.
3. **Lifecycle vocabulary only:** `ACTIVE`, `FAULTED`, `FINALIZED`, and `CLOSED`
   are declared states, not supported transition operations.

## Consequences

### Positive

- Later GPU execution layers can require shared dimensions and device identity.
- CPU-only metadata construction and package imports do not load Warp.
- Validation cost remains independent of resident payload size.

### Negative

- The boundary provides no scheduling, resource cleanup, recovery, or execution.
- Direct users must understand its concrete-only, unexported status.

### Neutral

- Existing Warp schemas, conversion helpers, direct kernels, and package exports
  remain unchanged.

## Implementation

### Required Changes

1. **Concrete session module** (`particula/execution/gpu_session.py`)
   - Retain compatible caller-owned resident containers and CPU metadata by
     identity after fixed-cost validation.
   - Lazily load Warp and generated types only during resident validation.
2. **Focused tests** (`particula/execution/tests/gpu_session_test.py`)
   - Cover import isolation, identity retention, schema rejection, immutable
     carriers, and no-transfer/no-synchronization construction behavior.
3. **Architecture documentation**
   - Record the concrete-only P1 scope and deferred operational semantics.

### Testing Strategy

Use focused CPU-only and Warp-optional tests. Warp cases use tiny fixtures and
verify metadata-only construction; CPU-only import checks verify no eager Warp or
`particula.gpu` import.

### Rollback Plan

Remove the direct module and tests if its invariant is replaced. No package
export, data migration, persistent resource, or operational lifecycle exists to
unwind.

## Validation

### Success Criteria

- [x] The module is direct-import-only and unexported from package surfaces.
- [x] Construction retains resources and metadata by identity after read-only
  fixed-cost schema and device validation.
- [x] Construction does not inspect payload values or perform runtime work.

## References

- [ADR-003: Dependency-Neutral Execution Capability Vocabulary](ADR-003-dependency-neutral-execution-capabilities.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1484](https://github.com/Gorkowski/particula/issues/1484)

## Notes

This ADR complements ADR-003 and supersedes none. P4 owns lifecycle operations
and transition guards; P5/P6 own finalization and failure/close behavior.
