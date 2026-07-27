# Architecture Design

## High-Level Design

Backend selection belongs in a separate `particula.execution` layer. Strategies
continue to describe physics, builders continue to construct strategies, and
`RunnableABC` remains the CPU `Aerosol` contract. The execution context resolves
an explicit request against immutable capability declarations and invokes a
typed adapter only after validation.

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
                     adapter selected explicitly
                                |
               +----------------+----------------+
               |                                 |
       CPUExecutionAdapter              future E7 GPU adapter
       RunnableABC + Aerosol             Warp resident state
               |                                 |
               +------> ExecutionResult <--------+
                       (state identity,
                        mutation metadata)
```

Selection never catches runtime failures to retry another backend. The neutral
module must import and execute in environments without Warp. Device resolution,
transfers, synchronization, and backend availability remain explicit adapter or
E7-F6 responsibilities.

## Data / API / Workflow Changes

- **Data model:** Add immutable typed declarations for backend, device request,
  process identity, capability set, execution request, and execution result.
  Define protocols for execution state and adapters. Do not modify CPU or Warp
  container schemas.
- **Capability matrix:** Key support by backend, process, and constraints;
  expose deterministic `supports()` and validating `require()` behavior.
  Declarations describe support but do not load kernels or probe by execution.
- **API surface:** Add `particula.execution` and deliberately re-export only the
  user-facing request/context/types from `particula`. Keep concrete adapters and
  registries module-local unless downstream extension requires a documented
  protocol.
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
