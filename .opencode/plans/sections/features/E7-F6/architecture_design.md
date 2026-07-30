# Architecture Design

## High-Level Design

E7-F6 extends E7-F1's dependency-neutral execution context. Capability and
availability resolution is read-only and occurs before adapter selection. A
typed fallback policy defaults to `ERROR`; an explicit CPU option may select the
CPU adapter only while CPU state is authoritative. Runtime exceptions are never
interpreted as capability misses.

```text
ExecutionRequest(requested=gpu, fallback=ERROR|EXPLICIT_CPU)
                         |
                         v
       capability + runtime/device availability preflight
               / supported             \ unavailable
              v                         v
      select GPU adapter       policy is ERROR? -> typed error
              |                         |
              |             EXPLICIT_CPU + CPU state authoritative?
              |                    / yes          \ no
              |                   v                v
              |          select CPU adapter   fallback-boundary error
              +-------------------+
                                  v
              ExecutionResult(requested_backend, selected_backend,
                              fallback_reason, mutation metadata)

After adapter invocation starts: propagate every failure; never retry.
Resident GPU state: checkpoint/finalize explicitly before a new CPU request.
```

## Data / API / Workflow Changes

- **P1 data model (implemented):**
  `particula.execution.errors` directly defines a standard-library-only
  `ExecutionCapabilityReason` enum, `ExecutionCapabilityError`, three category
  bases, and eight concrete fixed-reason errors. The root stores typed optional
  string context for backend, device, process, capability, state, and fallback
  boundary; deterministic rendering does not inspect exception chaining.
  The module is deliberately not exported from `particula.execution` or the
  top-level package.
- **Later data model:** Add a typed fallback policy and resolution metadata.
  Extend E7-F1 request/result only as needed to retain requested versus selected
  backend and fallback reason. Do not change scientific containers.
- **API surface:** Export user-actionable errors and fallback policy from
  `particula.execution` in P4 or later. Re-export only E7-F1-approved
  high-level execution names from `particula`; do not re-export concrete
  adapters, registries, sidecars, kernel configurations, or direct steps.
- **Availability:** CPU availability is dependency-neutral. GPU availability is
  provided lazily by the GPU integration layer and distinguishes missing Warp,
  unavailable device, and unsupported process. Checks must not launch physics.
- **Fallback boundary:** Explicit CPU fallback is selection, not exception
  recovery. It occurs before upload/mutation or after explicit restore. It does
  not call conversion/synchronization helpers itself.
- **Stability:** High-level execution request/error/policy contracts are stable
  once shipped. Existing low-level `particula.gpu.*` APIs remain callable but
  are documented experimental until E7-F9 full-loop validation. Breaking
  changes require release-note documentation and a deprecation path where
  technically feasible; this feature performs no removals.
- **Workflow hooks:** E7-F2/F3 adapters and E7-F4 sessions consume this resolver;
  E7-F5 scheduling propagates these errors and never inserts fallback.

## Security & Compliance

No permissions or network access change. Validation rejects untrusted backend,
device, and process identifiers rather than dynamically importing arbitrary
modules. Errors may report identifiers and reason codes but must not expose
array contents. Read-only preflight and fail-closed defaults protect state from
partial mutation and resource surprises.
