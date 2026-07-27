# Architecture Design

## High-Level Design

Backend selection belongs in a separate `particula.execution` layer. Strategies
continue to describe physics, builders continue to construct strategies, and
`RunnableABC` remains the CPU `Aerosol` contract. The execution context resolves
an explicit request against immutable capability declarations and returns a
typed adapter by identity; P2 does not invoke it.

```text
ExecutionRequest(backend, device, process, required capabilities)
                         |
                         v
                 CapabilityMatrix
                  / validate \
           unsupported      supported
             -> raise           |
                                v
                        ExecutionContext
                                 |
                   exact adapter selected (P2)
                                |
               +----------------+----------------+
               |                                 |
        future CPU adapter               future E7 GPU adapter
        RunnableABC + Aerosol             Warp resident state
```

Selection never catches runtime failures to retry another backend. The neutral
module must import and execute in environments without Warp. Device resolution,
transfers, synchronization, and backend availability remain explicit adapter or
E7-F6 responsibilities.

## P1 Implementation Record

Issue #1462 implements the declaration layer at `particula.execution` using
only the standard library. `CapabilityMatrix` is frozen and validates exact
typed declarations. Its nonempty lookup is membership-based, so separately
declared capabilities are never inferred as a combined capability; its empty
requirement lookup recognizes only a declared device/process base. `require()`
is a fail-closed pure wrapper around that lookup. No request/context, adapter,
registry, availability probe, transfer, execution path, or public package
 export was added.

## P2 Implementation Record

Issue #1463 adds selection only. `ExecutionRequest` validates fields in
backend/device/process/requirements order and rejects mismatched backend/device
pairs at construction. `ExecutionContext.resolve()` validates and normalizes a
request, invokes immutable-matrix `require()`, and then performs one private
exact `(Process, Backend)` lookup. Each context owns its own registry; malformed
or duplicate registrations do not replace entries. CPU accepts only the
canonical `Device(Backend.CPU, "cpu")`; a Warp device's native value is passed
unchanged to matrix validation. No adapter call, fallback, retry, transfer,
optional-backend import, availability probe, state/result contract, or package
 export is part of this phase.

## P3 Implementation Record

Issue #1464 adds contracts only, beside the P2 selection seam. Runtime-checkable
`ExecutionState` requires only an opaque readable `backend_payload`, while
`ExecutionAdapter` documents the future execute seam without changing P2's
callable-only registration rule. `MutationDeclaration` accepts exactly one of
the closed `NONE` or `STATE` scopes. Frozen `ExecutionResult` retains its state,
ordered immutable metadata, declaration, and optional opaque `BackendResult`.
`validate_execution_result()` validates the original structural state, exact
result layout, retained state identity, metadata, declaration, and wrapper type
in order, returning the same result object. It neither invokes adapters nor
inspects opaque fields, imports a backend, transfers data, or adds fallback.

## Data / API / Workflow Changes

- **Data model:** Immutable typed declarations now cover backend, device request,
  process identity, capability set, execution request, and internal P3 result
  contracts. P3 protocols preserve opaque state ownership and explicit mutation
  permission without modifying CPU or Warp container schemas.
- **Capability matrix:** Key support by backend, process, and constraints;
  expose deterministic `supports()` and validating `require()` behavior.
  Declarations describe support but do not load kernels or probe by execution.
- **API surface:** P3 is confined to `particula.execution`; it adds no top-level
  export. Concrete adapters and registries remain module-local, and publication
  remains a later deliberate phase.
- **CPU adapter:** Accept CPU state carrying an `Aerosol`, delegate exact
  `time_step`/`sub_steps` to `RunnableABC.execute()`, retain the returned object,
  and report in-place mutation semantics. It does no conversion or fallback.
- **Workflow hooks:** E7-F6 extends validation with availability/error policy;
  E7-F2/F3 register process adapters; E7-F4 supplies resident session state;
  E7-F5 consumes contexts in deterministic scheduling.
- **Compatibility:** Existing direct CPU and GPU APIs remain callable and
  unchanged. This is an additive boundary, not a replacement or physics rewrite.

## Security & Compliance

No credentials, network permissions, or regulated data are introduced. Robust
validation is safety-relevant: reject unknown adapters, invalid devices,
capability mismatches, and malformed state before mutation. Never deserialize
or dynamically import an adapter from an untrusted string. Avoid broad exports
of concrete scratch/configuration internals and do not hide device transfers or
fallback behind exception handling.
