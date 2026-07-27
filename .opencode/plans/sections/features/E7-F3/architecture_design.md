# Architecture Design

## High-Level Design

E7-F3 is a process adapter between E7-F1's dependency-neutral execution context
and the existing CPU and direct-Warp Brownian implementations. E7-F6 policy
must validate backend/device availability and prohibit implicit fallback before
adapter dispatch. The adapter owns mapping and result semantics, while each
concrete implementation remains authoritative for physics and detailed runtime
validation.

```text
ExecutionRequest(process=coagulation, mode=brownian, backend, device)
                              |
                 CapabilityMatrix.require (E7-F1/F6)
                              |
                    BrownianCoagulationAdapter
                     /                         \
      CPU state + Coagulation             Warp resident state
      Brownian strategy                    particles/environment
      exact sub_steps                      outputs + rng_states
                     \                         /
                      one explicit backend call
                              |
                     ExecutionResult
          particle/state identity + collision diagnostics
                 + mutation/RNG progression metadata
```

The Warp state references caller-owned particles, optional collision outputs,
and a required persistent RNG resource for selected repeated-step use. Setup
seeds the buffer once; normal calls pass `initialize_rng=False`; explicit reset
is never inferred from a repeated seed value. The adapter must not upload,
restore, synchronize, allocate a hidden long-lived stream, or catch a runtime
failure to invoke CPU.

## Data / API / Workflow Changes

- **Data model:** Add immutable Brownian process configuration and typed CPU/Warp
  state views. The Warp view carries particle/environment or direct physical
  inputs, optional volume and collision outputs, max-collision policy, RNG seed,
  caller-owned `(n_boxes,)` RNG state, and explicit reset intent. Existing CPU
  and Warp container schemas do not change.
- **Capability mapping:** CPU admits the existing Brownian strategy modes that
  E7-F1 declares. Warp selection is intentionally limited to shipped Brownian,
  particle-resolved execution. Other direct-kernel mechanisms are not promoted
  by T3 and fail as capability errors.
- **API surface:** Register a Brownian coagulation adapter through E7-F1's typed
  registry. Export only stable configuration/state names allowed by E7-F6.
  Keep direct kernel configuration and scratch internals concrete-only.
- **CPU behavior:** Delegate to `Coagulation.execute()` with exact time and
  substeps and preserve its returned `Aerosol` semantics. Do not promise shared
  stochastic trajectories with Warp.
- **Warp behavior:** Delegate once per selected step to `coagulation_step_gpu`
  using explicit Brownian configuration. Return the identical particles and
  supplied collision buffers through `ExecutionResult`; report accepted counts
  without host readback. RNG mutates in place but is not added to the kernel's
  return tuple.
- **Workflow hooks:** E7-F4 may store output/RNG resources in resident session
  state. E7-F5 schedules this adapter. E7-F8 extends this seam with stream
  identity, box independence, and checkpoint/restart semantics.

## Validation and Failure Semantics

Selection-level checks cover request/process/mode, backend/device availability,
state kind, finite nonnegative time, explicit Brownian-only capability, and
required persistent resource declarations before adapter invocation. The Warp
entry point remains authoritative for particle arrays, environment/volume,
output capacity/dtype/device, RNG schema, and detailed validation order.
Pre-launch rejection must leave caller state unchanged. After Warp launches,
particles, outputs, or RNG may be partially changed and no rollback is promised.

## Security & Compliance

No credentials, network access, or regulated data are introduced. Fail-closed
typed registration avoids loading adapters from untrusted strings. Explicit
device/resource validation prevents accidental cross-device access, and bounded
exports prevent experimental kernel internals from becoming accidental public
API. Resource exhaustion is bounded by fixed particle and collision capacities.
