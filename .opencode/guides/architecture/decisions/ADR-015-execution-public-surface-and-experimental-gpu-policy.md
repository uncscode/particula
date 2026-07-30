# ADR-015: Execution Public Surface and Experimental GPU Policy

**Status:** Accepted  
**Date:** 2026-07-30  
**Decision Makers:** ADW Development Team  
**Technical Story:** [#1503](https://github.com/Gorkowski/particula/issues/1503)

## Context

The dependency-neutral execution package already exposes stable selection
metadata, while concrete execution mechanics remain intentionally separated by
direct-import boundaries. The prior execution and fallback ADRs did not record
the new public policy for capability errors and fallback policy values. The
low-level GPU container, explicit-transfer, and direct-kernel workflow also
needs a clear stability designation without changing its import or execution
semantics.

### Problem Statement

Freeze a narrow, ordered public execution surface without accidentally promoting
concrete mechanics, and describe the existing low-level GPU workflow as
experimental without changing it.

### Forces

**Driving Forces:**
- Callers need a stable capability-error taxonomy and fallback-policy vocabulary.
- CPU-only imports must remain free of Warp and GPU-package dependencies.
- Explicit GPU ownership and transfer boundaries must remain visible.

**Restraining Forces:**
- Re-exporting implementation carriers or operations would broaden lifecycle,
  fallback, and backend commitments.
- A stability label must not introduce warnings, eager loading, or semantic
  changes to supported direct imports.

## Decision

Freeze the ordered 26-name `particula.execution` surface and re-export that
same execution subset from top-level `particula`:

1. Ten selection declarations: `Backend` through `ExecutionContext`.
2. Thirteen closed capability errors: `ExecutionCapabilityReason` through
   `FallbackDisallowedError`.
3. Three fallback policy enums: `FallbackPolicy`, `FallbackBoundary`, and
   `CPUStateAuthority`.

The concrete `errors` and `fallback` modules, fallback operations and carriers,
availability, adapters, state/result carriers, registries, sessions, sidecars,
and GPU mechanics remain direct-import-only. `particula.gpu` low-level
containers, explicit transfers, and direct-kernel workflow are experimental;
their current import paths and caller-owned transfer model are unchanged.

### Chosen Option

**Option 2: Narrow value re-exports with concrete direct-import boundaries**

Re-export only the closed vocabulary needed by callers. Keep modules and all
operational or resource-owning types concrete, and communicate the GPU
experimental policy in documentation and its module description only.

## Alternatives Considered

### Option 1: Keep the original ten-name execution surface

**Pros:** No public-surface expansion.

**Cons:** Callers must import the concrete errors and fallback modules merely to
handle public selection failures or choose a policy.

**Reason for Rejection:** The error taxonomy and policy enums are stable public
vocabulary, unlike the mechanics that use them.

---

### Option 2: Narrow value re-exports with concrete direct-import boundaries (chosen)

**Pros:** Provides a complete caller-facing vocabulary while preserving
dependency-neutral imports and concrete ownership boundaries.

**Cons:** The ordered allowlist must be maintained deliberately.

**Reason for Selection:** It exposes stable decisions without promoting
operations, carriers, or backend mechanics.

---

### Option 3: Re-export execution and GPU implementation modules broadly

**Pros:** Shorter import paths for implementation types.

**Cons:** Commits lifecycle, fallback, resource, and experimental GPU details to
a broad compatibility surface.

**Reason for Rejection:** Those types require direct imports to preserve their
bounded contracts.

## Rationale

Public error values and policy enums are dependency-neutral value vocabulary;
their concrete modules and operations are not. An exact ordered allowlist makes
that distinction testable at both execution and top-level boundaries. Marking
the GPU workflow experimental communicates its maturity without hiding the
explicit transfer model or changing warning, loading, export, or runtime
behavior.

### Trade-offs Accepted

1. **Explicit maintenance:** Any execution export change requires an intentional
   compatibility decision and allowlist update.
2. **Concrete imports remain:** Advanced callers must continue to import
   operations, carriers, and GPU mechanics from their defining modules.

## Consequences

### Positive

- Capability failures and fallback policy choices have stable public imports.
- Concrete execution and GPU ownership boundaries remain unambiguous.
- The CPU-only execution import remains independent of optional GPU runtime
  loading.

### Negative

- The experimental GPU workflow offers no new stability guarantee.
- The public surface deliberately does not shorten imports for implementation
  mechanics.

### Neutral

- Existing direct GPU imports, explicit transfers, and caller-owned state
  semantics are unchanged.

## Implementation

### Required Changes

1. **Execution exports** (`particula/execution/__init__.py` and
   `particula/__init__.py`)
   - Publish the exact ordered 26-name execution value surface.
   - Keep concrete modules, operations, and carriers unexported.
2. **Boundary documentation** (`.opencode/guides/architecture/`)
   - Record the public-value versus concrete-mechanics distinction.
   - Mark the low-level GPU workflow experimental without changing its contract.

### Testing Strategy

Verify the ordered execution allowlist and top-level identities in a fresh
CPU-only process; deny concrete names at both public boundaries; and retain
warning-free GPU imports with their existing exports and lazy behavior.

### Rollback Plan

Remove the additional re-exports and restore the prior allowlist only through a
new compatibility decision. No concrete execution or GPU implementation needs
to be moved to roll back this policy.

## Validation

### Success Criteria

- [x] The execution and top-level boundaries expose the same ordered 26-name
  execution subset by identity.
- [x] Concrete modules, operations, carriers, lifecycle seams, and GPU mechanics
  remain absent from those public execution surfaces.
- [x] CPU-only execution import does not load Warp or `particula.gpu`.
- [x] GPU experimental wording does not alter direct imports, exports, warnings,
  or caller-owned transfer semantics.

## References

- [ADR-003: Dependency-Neutral Execution Capability Vocabulary](ADR-003-dependency-neutral-execution-capabilities.md)
- [ADR-014: Opt-In CPU Fallback Boundary](ADR-014-opt-in-cpu-fallback-boundary.md)
- [Architecture Guide](../architecture_guide.md)
- [Architecture Outline](../architecture_outline.md)
- [Issue #1503](https://github.com/Gorkowski/particula/issues/1503)

## Notes

This ADR amends the export-policy portions of ADR-003 and ADR-014; it does not
supersede their selection or fallback-mechanics decisions. No prior ADR is
archived.
