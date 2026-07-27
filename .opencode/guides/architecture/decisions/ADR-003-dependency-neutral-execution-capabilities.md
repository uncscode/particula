# ADR-003: Dependency-Neutral Execution Capability Vocabulary

**Status:** Accepted
**Date:** 2026-07-27
**Decision Makers:** ADW Development Team
**Technical Story:** [#1462](https://github.com/Gorkowski/particula/issues/1462)

## Context

Future execution work needs a common way to declare which backend, opaque
device, process, and exact capability combinations are supported. That
vocabulary must be usable without importing an optional execution backend or
prematurely committing to adapter, context, transfer, or execution behavior.

### Problem Statement

Provide a typed, immutable capability declaration boundary that can represent
support without probing availability, resolving devices, selecting an adapter,
or running a process.

### Forces

**Driving Forces:**
- CPU and future Warp-backed paths need comparable support declarations.
- Optional-backend imports must remain outside dependency-neutral metadata.
- Capability declarations must not silently compose separate entries.

**Restraining Forces:**
- Device-native identifiers have backend-specific semantics that P1 must not
  interpret.
- Contexts, requests, registries, adapters, data transfer, and execution need
  later lifecycle decisions.

## Decision

Add the deliberately dependency-neutral `particula.execution` P1 module. It
provides immutable standard-library-only `Backend`, `Device`, `Process`,
`Capability`, requirement/declaration records, and a pure exact-match
`CapabilityMatrix`. The module is not exported from top-level `particula`.

### Chosen Option

**Option 2: Typed declaration-only capability vocabulary**

The module will:

1. Preserve a `Device` native identifier as opaque metadata.
2. Use a frozen matrix with exact whole-declaration matching for nonempty
   requirements and a declared-base rule for empty requirements.
3. Avoid imports of Warp, `particula.gpu`, or any optional backend.
4. Leave execution contexts, requests, adapters, registries, availability
   probing, device resolution, data transfer, and execution to P2 and later.

## Alternatives Considered

### Option 1: Put capability metadata in `particula.gpu`

**Pros:** Places Warp-related declarations beside existing GPU code.

**Cons:** Couples general execution metadata to an optional backend and makes
CPU-only imports depend on the GPU package boundary.

**Reason for Rejection:** Capability declarations must precede and remain
independent of backend-specific implementation choices.

---

### Option 2: Typed declaration-only capability vocabulary (chosen)

**Pros:** Establishes a stable, immutable vocabulary with no optional runtime
dependency or execution side effects.

**Cons:** Does not itself choose or execute a supported implementation.

**Reason for Selection:** It creates the required P1 boundary without
prejudging future execution lifecycle design.

---

### Option 3: Add a registry that resolves and executes immediately

**Pros:** Could provide a single high-level entry point.

**Cons:** Mixes static support declarations with backend availability, adapter
selection, context ownership, and data movement.

**Reason for Rejection:** These are P2+ responsibilities requiring separate
contracts and lifecycle decisions.

## Rationale

Frozen typed value objects make declarations structural and safely reusable.
Exact matching prevents separately declared capabilities from being inferred as
a combined supported configuration. Keeping native device identifiers opaque
avoids duplicating backend-specific validation or availability behavior.

### Trade-offs Accepted

1. **Declaration only:** A positive capability result does not prove runtime
   availability or execute a process.
2. **Closed initial backend vocabulary:** P1 names CPU and Warp only; future
   expansion requires an intentional compatibility decision.

## Consequences

### Positive

- Importing capability metadata has no Warp or GPU dependency.
- Future execution layers receive one typed, exact-match support vocabulary.
- Unsupported combinations fail closed rather than composing declarations.

### Negative

- Callers cannot resolve native devices or run work through this module.
- P2+ must add contexts, requests, adapters, and registries separately.

### Neutral

- `particula.execution` remains a direct import and is not a top-level package
  export.

## Implementation

### Required Changes

1. **Metadata module** (`particula/execution.py`)
   - Define immutable declarations and pure `CapabilityMatrix` lookup.
   - Restrict imports to the Python standard library.
2. **Focused coverage** (`particula/tests/execution_test.py`)
   - Verify exact matching, immutability, validation order, and a guarded fresh
     import that rejects Warp and GPU imports.

### Testing Strategy

Run the focused execution tests with warnings as errors, capability-module
coverage, and Ruff. Verify that guarded fresh imports do not load `warp` or
`particula.gpu`.

### Rollback Plan

Remove the isolated module and its tests. No top-level exports, backend
execution paths, or runtime state depend on P1.

## Validation

### Success Criteria

- [x] The module contains immutable standard-library-only declarations and
  matrix lookup.
- [x] Matrix lookup uses whole exact declarations and does not compose
  capabilities.
- [x] The module does not resolve devices, probe availability, transfer data,
  select adapters, or execute work.
- [x] The module is not exported from top-level `particula`.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1462](https://github.com/Gorkowski/particula/issues/1462)

## Notes

No prior ADR is superseded. P2+ owns execution contexts, requests, adapters,
and registries.
