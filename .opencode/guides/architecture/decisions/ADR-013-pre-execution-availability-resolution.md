# ADR-013: Pre-Execution Availability Resolution

**Status:** Accepted
**Date:** 2026-07-30
**Decision Makers:** ADW Development Team
**Technical Story:** [#1501](https://github.com/Gorkowski/particula/issues/1501)

## Context

The dependency-neutral P1 execution vocabulary declares support, and P2
selection resolves only a registered adapter. Neither establishes that an
optional runtime is installed, a declared native device can be used, or
request-associated state is valid. Downstream resident setup needs this check
without coupling P1 metadata or adapter selection to Warp imports, data motion,
or execution.

### Problem Statement

Provide one deterministic, fail-closed availability boundary for validated P1
metadata before execution while preserving opaque native-device identifiers and
the existing package export boundary.

### Forces

**Driving Forces:**
- CPU and Warp requests require comparable pre-execution failure semantics.
- Optional Warp imports must remain lazy and absent from CPU-only resolution.
- Resident setup requires an availability-approved native device without
  independently probing or selecting it.

**Restraining Forces:**
- P1 metadata and package-level selection must remain dependency-neutral.
- Native device strings are backend-owned and must not be parsed or normalized.
- Availability must not imply adapter ownership, data ownership, or permission
  to execute work.

## Decision

Add `particula.execution.availability` as a concrete, direct-import-only P2
resolver. `resolve_availability()` accepts validated `ExecutionRequest` and
`CapabilityMatrix` metadata and returns a frozen `AvailabilityDecision` that
retains only the exact request.

The resolver will:

1. Validate an injected or default provider registry as exactly usable CPU and
   Warp providers before provider methods or optional runtime work.
2. Short-circuit in fixed order: recognition, structural process declaration,
   exact capability declaration, lazy runtime status, device status, then
   injected request-associated state validation.
3. Recognize only `Device(Backend.CPU, "cpu")` for CPU; recognize every
   validated Warp declaration without interpreting its native string, then pass
   that unchanged string to lazy Warp device resolution.
4. Map false, malformed, and exceptional phase results to the existing typed
   availability errors and retain no adapter, runtime, device, payload, or state
   object in the successful decision.

The module remains absent from `particula.execution` and top-level exports. It
does not select or invoke adapters, fall back, allocate, transfer, synchronize,
mutate, or launch work.

### Chosen Option

**Option 2: Separate direct-import availability resolver**

Keep runtime and device checks in a narrow concrete module downstream of P1
metadata and separate from adapter selection and resident setup.

## Alternatives Considered

### Option 1: Add availability checks to package-level selection

**Pros:** One apparent entry point for support and availability.

**Cons:** Makes the dependency-neutral selection seam import or probe optional
runtimes and conflates static declarations with runtime status.

**Reason for Rejection:** P1/P2 selection intentionally has no optional-runtime
or availability behavior.

---

### Option 2: Separate direct-import availability resolver (chosen)

**Pros:** Establishes deterministic fail-closed ordering while retaining lazy
optional-runtime handling and the existing export boundary.

**Cons:** Callers needing availability must import a concrete module directly.

**Reason for Selection:** It supplies the required native-availability
precondition without broadening selection or resident-session ownership.

---

### Option 3: Let each execution or setup boundary probe availability

**Pros:** Avoids another module.

**Cons:** Duplicates provider policy, creates inconsistent error ordering, and
risks parsing native strings or adding hidden runtime work.

**Reason for Rejection:** Availability policy is cross-cutting and must be
centralized before downstream execution boundaries.

## Rationale

Separating structural declarations from lazy runtime checks preserves the P1
contract while making failures deterministic. A complete fixed provider registry
and exact boolean status results fail closed before any later phase. Treating
Warp native identifiers as opaque lets the optional runtime remain authoritative
for device interpretation; the canonical CPU declaration needs no probe.

### Trade-offs Accepted

1. **Concrete import:** Availability is intentionally not part of the stable
   package-level selection API.
2. **No fallback:** A failed provider phase reports its typed failure and never
   chooses another backend or device.
3. **Request-only decision:** Success proves only the declared preconditions;
   it does not create a runtime handle or authorize execution.

## Consequences

### Positive

- CPU-only calls remain optional-runtime-neutral.
- Downstream setup can require one consistent availability precondition.
- Failure order and typed error mapping are deterministic and testable.

### Negative

- Availability callers must supply complete provider policy when overriding
  defaults.
- A successful decision must still be followed by the owning adapter or setup
  boundary's local validation.

### Neutral

- P1 metadata, package-level adapter selection, direct GPU APIs, and resident
  lifecycle ownership remain unchanged.

## Implementation

### Required Changes

1. **Availability resolver** (`particula/execution/availability.py`)
   - Define the provider and state-validator contracts plus frozen request-only
     decision record.
   - Implement default CPU and lazy Warp providers with fixed validation order.
2. **Contract tests** (`particula/execution/tests/availability_test.py`)
   - Cover ordering, failure mappings, opaque Warp device handling, and guarded
     CPU-only imports.
3. **Architecture documentation**
   - Record this concrete boundary, exclusions, and ADR index.

### Testing Strategy

Test default CPU and injected providers without Warp/CUDA, all short-circuit
failures and typed contexts, lazy Warp import and opaque-string forwarding, and
the guarded optional-runtime-neutral import path. Run focused execution tests
with warnings as errors plus Ruff and mypy checks.

### Rollback Plan

Remove the isolated concrete module and its tests. No package export, adapter
registry, resident container schema, or direct-kernel API depends on its
successful decision object.

## Validation

### Success Criteria

- [x] Registry validation precedes all provider and optional-runtime calls.
- [x] Resolution follows the specified six-phase short-circuit order.
- [x] CPU is canonical-only and Warp native strings remain opaque until lazy
  device resolution.
- [x] The decision retains only the request and introduces no execution or data
  movement behavior.
- [x] The concrete module is not package- or top-level-exported.

## References

- [ADR-003: Dependency-Neutral Execution Capability Vocabulary](ADR-003-dependency-neutral-execution-capabilities.md)
- [ADR-005: One-Time GPU Resident-Session Setup](ADR-005-one-time-gpu-resident-session-setup.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1501](https://github.com/Gorkowski/particula/issues/1501)

## Notes

This ADR extends the execution architecture and supersedes none.
