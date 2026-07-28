# ADR-003: Dependency-Neutral Execution Capability Vocabulary

**Status:** Accepted (amended for shipped P2--P5 surface and #1471 package
migration)
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

Add the deliberately dependency-neutral `particula.execution` P1 package. It
provides immutable standard-library-only `Backend`, `Device`, `Process`,
`Capability`, requirement/declaration records, and a pure exact-match
`CapabilityMatrix`. The module remains a direct import rather than a top-level
`particula` export.

### Amendment: Shipped P2--P5 Surface

P2--P5 are now shipped on the same dependency-neutral module boundary. P2 adds
`ExecutionRequest`, `ExecutionAdapter`, and context-local `ExecutionContext`
registration and exact selection. Registration and selection inspect only the
callable seam and capability declarations; they do not invoke adapters, probe
availability, resolve devices, transfer state, or provide fallback.

P3 adds direct-module-only `ExecutionState`, `MutationDeclaration`,
`BackendResult`, `ExecutionResult`, and result validation. These carriers retain
caller-owned opaque payloads and state by identity. P4 adds the direct-module-
only `CPUExecutionState` and `CPUExecutionAdapter`, a narrow CPU runnable
dispatch boundary. P5 deliberately exports only the stable selection vocabulary
through `particula.execution`; P3/P4 carriers and CPU dispatch remain excluded
from the top-level package and are not a high-level execution API.

### Amendment: #1471 Package Migration and Condensation P2 Carriers

`particula.execution` is now a package, preserving its exact ten-name public
selection export surface. Its concrete-only
`particula.execution.adapters.condensation` module supplies P2 configuration
and CPU/Warp state carriers plus shipped P3/P4 selected adapters, without
promoting them through `particula.execution` or top-level `particula`. The
package-level selection seam remains dependency-neutral; the concrete module
imports Warp only when validating a Warp state or selected sidecars. Carrier
construction retains caller-owned resources by identity and performs read-only
metadata/ownership validation. After exact profile preflight, P3 CPU dispatches
one supplied isothermal runnable call and P4 Warp dispatches one direct-kernel
call; neither transfers, restores, synchronizes, falls back, nor recovers
failures. Native property-array fallback allocation remains a direct-kernel
contract, while callers own reusable sidecars and post-launch synchronization.

### Chosen Option

**Option 2: Typed declaration-only capability vocabulary**

The module will:

1. Preserve a `Device` native identifier as opaque metadata.
2. Use a frozen matrix with exact whole-declaration matching for nonempty
   requirements and a declared-base rule for empty requirements.
3. Avoid imports of Warp, `particula.gpu`, or any optional backend.
4. At P1 adoption, leave execution contexts, requests, adapters, registries,
   availability probing, device resolution, data transfer, and execution to P2
   and later.

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

- The module does not resolve native devices, probe runtime availability, or
  provide transfer, fallback, scheduler, or high-level GPU-adapter behavior.
- P2/P3/P4 contracts remain narrowly separated; selection does not execute,
  P2 validates ownership, and the concrete P3/P4 adapters provide only their
  one-call CPU/Warp condensation dispatch boundaries.

### Neutral

- `particula.execution` remains a direct import and is not a top-level package
  export.

## Implementation

### Required Changes

1. **Selection package** (`particula/execution/__init__.py`)
   - Define immutable declarations and pure `CapabilityMatrix` lookup.
   - Restrict imports to the Python standard library.
2. **Concrete condensation carriers**
   (`particula/execution/adapters/condensation.py`)
   - Keep construction-only P2 carriers unexported and retain resources by
     identity.
   - Defer the optional Warp import until Warp-state validation.
3. **Focused coverage** (`particula/tests/execution_test.py` and
   `particula/execution/tests/`)
   - Verify exact matching, immutability, validation order, and a guarded fresh
   import that rejects Warp and GPU imports.

### Testing Strategy

Run the focused execution tests with warnings as errors, capability-module
coverage, and Ruff. Verify that guarded fresh imports do not load `warp` or
`particula.gpu`.

### Rollback Plan

Replace the isolated package and its tests only with a replacement for the
shipped selection and concrete-module execution contracts. The public selection
API has no optional-backend dependency, transfer path, or high-level execution
integration to unwind.

## Validation

### Success Criteria

- [x] The module contains immutable standard-library-only declarations and
  matrix lookup.
- [x] Matrix lookup uses whole exact declarations and does not compose
  capabilities.
- [x] The P1 metadata layer does not resolve devices, probe availability,
  transfer data, select adapters, or execute work.
- [x] The shipped P2 layer performs exact context-local selection without
  adapter execution; P3 ownership validation and P4 CPU dispatch retain their
  documented direct-module-only boundaries.
- [x] The module is not exported from top-level `particula`.

## References

- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1462](https://github.com/Gorkowski/particula/issues/1462)
- [Issue #1471](https://github.com/Gorkowski/particula/issues/1471)

## Notes

No prior ADR is superseded. This ADR records the original P1 decision and its
P2--P5 amendment; future availability, fallback, GPU-adapter, transfer, and
scheduling decisions remain separate Epic G work. The concrete P1
resident-session boundary is recorded separately in
[ADR-004](ADR-004-concrete-gpu-resident-session-boundary.md).
