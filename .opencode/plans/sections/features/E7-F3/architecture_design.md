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

### P2 Carrier Boundary (implemented through P4)

`particula.execution.adapters.coagulation` now supplies concrete-only frozen
CPU and resident-Warp Brownian request/result carriers. They retain resources
by identity, use an exact `BrownianCoagulationConfig` marker, selection-owned
kind/form, selected-time, and metadata-detectable alias checks, and leave
physical schemas, direct-kernel validation, dispatch, and mutation to the
native boundary. The marker check precedes optional Warp import and lazy-kernel
resolution, so request-shaped non-Brownian values fail closed without probing
the native resolver. `rng_states`, `rng_seed`, and `initialize_rng` record
caller-owned persistent-RNG intent without seeding, resetting, advancing, or
otherwise mutating the sidecar. At P2, the carriers did not dispatch or import
kernels for invalid markers; they performed no transfer, synchronization,
allocation, or export change. The direct Warp kernel remains authoritative for
particle, environment/volume, output-buffer, device, dtype, capacity, and
detailed RNG schemas.

### P3 Dispatch Boundary (implemented)

`CPUCoagulationExecutionState` retains an exact P2 CPU state, opaque controls,
and an exact `Coagulation` runnable. Its adapter validates finite,
nonnegative real `time_step` and positive integral `sub_steps` before making
one positional `Coagulation.execute(aerosol, time_step, sub_steps)` call. The
returned aerosol must retain the P2 aerosol identity.

`WarpBrownianCoagulationExecutionState` retains the exact P2 Warp state without
resolving the kernel or revalidating native resource schemas. Its adapter resolves
`coagulation_step_gpu` lazily only after exact-state preflight, calls it once
with the P2 direct/environment form, diagnostics, and RNG sidecar/intent, and
wraps its native tuple only after existing identity checks. Neither adapter
converts, synchronizes, retries, selects another backend, catches failures, or
rolls back delegate mutation. Both emit `ExecutionResult` with
`MutationScope.STATE` and a `BackendResult` containing the corresponding typed
P2 result. P1/E7-F6 capability policy and later runtime-mode validation remain
separate work.

### P5 Evidence Boundary (implemented)

`particula/execution/tests/coagulation_integration_test.py` exercises the
existing concrete-only adapter rather than extending its runtime boundary. CPU
Brownian rate/reference checks are separate from resident-Warp invariant and
stochastic evidence. Warp tests retain caller-owned particle, diagnostics, and
RNG resources by identity; synchronize only in tests before host observation;
and verify concentration-weighted mass, signed charge, valid active-only pairs,
inactive sentinel preservation, and box isolation. A fixed 100 fresh-trial
experiment checks aggregate acceptance, while separate nonterminal repeated-call
and explicit-reset cases establish persistent RNG behavior. CUDA is optional and
invariant-only. No adapter transfer, restore, synchronization, fallback, or
public/export change was introduced.

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

The concrete P4 selection boundary accepts only the exact Brownian marker before
CPU dispatch, optional Warp import, or Warp resolver access. Warp P2 validates
kind/form and finite, nonnegative selected time before required RNG and
metadata-detectable alias checks. CPU validates finite, nonnegative selected
time and positive integral `sub_steps` before its sole runnable call. The Warp
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
