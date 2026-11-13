# Architecture Design

## High-Level Design

Communication is an explicit, opt-in scheduler operation over authoritative
resident state. Acquisition validates P1 exactly once, then pins one complete
closed-map GAS or PARTICLES resource family by identity: configuration/map
arrays, the matching native buffer record, and optional final volumes. Normal
steps perform metadata/identity validation only.

```text
PrescribedCommunication + PrescribedVolumeUpdate
                    |
       one-time P1 acquisition/schema preflight
                    |
                    v
E7-F5 communication barrier inside E7-F4 mutation window
                    |
        stage pre-step extensive inventories
        amount = concentration * pre-step volume
                    |
       +------------+-------------+
       |                          |
 gas edge ledger          particle slot plan
 (species amounts)        (population + mass/charge metadata)
       +------------+-------------+
                    |
        validate conservation/capacity/status
                    |
         optional prescribed positive volume evolution
                    |
         commit once; mark saturation ratio stale only
                    |
   ten existing loop nodes and their consumer refresh windows
```

Independent boxes are represented by an empty/disabled map and remain bitwise
unmodified by the communication node. Closed maps conserve total gas amount,
particle number, concentration-weighted species mass, and
concentration-weighted charge within floating-point tolerances. Open boundaries
must declare source/sink ledgers so apparent non-conservation is explicit.

## Data / API / Workflow Changes

- **P1 declaration boundary (shipped):** Concrete-only
  `particula.execution.communication` provides frozen, identity-retaining map,
  transport-mode, resource-shape, optional final-volume, and configuration
  declarations. Its sole entry point validates fixed Warp schemas, metadata,
  ranges/aliases, finite domains, enabled topology, and duplicate directed
  edges without writes, payload copies, primary-state reads, or exports.
   Population-dependent outbound overdraw is explicitly deferred to P3, which
   alone has source inventory and `time_step` inputs.
- **P2 volume boundary (shipped, #1508):** Concrete-only
  `particula.gpu.kernels.communication.volume_evolution_step_gpu` accepts
  caller-owned same-device final `(B,)` `wp.float64` volumes in m³. After
  read-only validation of all primary storage, aliases, and physical domains,
  it sets `particles.volume` and scales particle and gas concentrations by
  `old_volume / final_volume` in place. It preserves containers, arrays, and
  protected fields by identity; equal-volume calls are write-free. It neither
  transfers/mixes payloads nor integrates with the resident session or
   scheduler, and has no export, transfer, fallback, resize, or rollback after
   an apply writer launches.
- **P3 gas boundary (shipped, #1509):** Concrete-only
  `particula.gpu.kernels.communication.gas_communication_step_gpu` accepts
  complete Warp particle/gas containers, an exact P1 GAS configuration, finite
  nonnegative scalar `time_step`, and caller-owned `GasCommunicationBuffers`
  (or the corresponding `(B, S)` float64 work/accounting arrays). It stages
  `amounts = gas.concentration * particles.volume`, computes every enabled
  transfer from that immutable ledger, atomically accumulates in-domain deltas
  and declared `-1` boundary source/sink accounting, rejects aggregate
  overdraw, and makes one gas-concentration commit. It validates optional
  prescribed-volume metadata but never writes it, particle fields, or volume.
  It remains unexported and has no scheduler/session integration, transfer,
   synchronization, fallback, resize, or post-launch rollback.
- **P4 particle boundary (shipped, #1510):** Concrete-only
  `particula.gpu.kernels.communication.particle_communication_step_gpu` accepts
  a complete Warp particle container, an exact P1 PARTICLES configuration, a
  finite nonnegative `time_step`, and caller-owned
  `ParticleCommunicationBuffers`. Its immutable pre-step planner aggregates
  source debits and destination credits, assigns each transferred population to
  an exact pre-step destination match or the next ascending pre-step free slot,
  and rejects overdraw or capacity shortage before the gated one-kernel commit.
  It supports closed in-domain maps only: it does not access gas, change volume,
  accept `-1` endpoints, resize, compact, transfer, synchronize, fall back, or
  integrate with the session or scheduler. The API and buffer carrier remain
  concrete-only and unexported.
- **Resident resources (P5, shipped):** `gpu_resources.py` exposes frozen
  `CommunicationResources` and pins exactly one validated closed GAS or
  PARTICLES family. `acquire_communication()` validates P1 once, validates or
  allocates omitted required work arrays, rejects aliases, and permits only the
  identical complete reacquisition. Metadata-only validation performs no P1
  scan, allocation, transfer, synchronization, or payload readback.
- **Checkpointing (P5, shipped):** controller-created checkpoints use schema 2.
  They contain no communication family or one complete family plus matching
  `CommunicationCheckpointMetadata`; restart recreates fresh configuration,
  arrays, native buffers, and binding. Schema-v1 remains valid only for
  noncommunication checkpoints and remains restart-compatible.
- **Graph/scheduler (P5, shipped):** `communication` then `volume_evolution`
  are the first two nodes of the canonical twelve-node graph. Both are
  non-process barriers and invalidate `SATURATION_RATIO` only, preserving fresh
  vapor pressure for the existing condensation and diagnostics refresh windows.
  Preflight failures leave the session reusable; a failure after either native
  writer is about to run closes the guard and faults the session without retry
  or rollback.
- **Data model:** The fixed edge capacity, source/destination
  `int32` arrays, enabled mask, finite nonnegative inverse-time rates,
  transport mode, and optional final volumes remain stable for the session.
- **API surface:** P1 declarations and validation are direct-import-only from
  `particula.execution.communication`; they are not exported through
  `particula.execution` or the package. The resident executor and scheduler
  request remain concrete-only direct imports.
- **Gas semantics:** Treat `gas.concentration` as mass per volume. P3 stages
  `concentration * volume`, transfers synchronously, accounts for open
  boundaries explicitly, and divides final amount by the unchanged validated
  current volume. Volume evolution is the separate P2 operation.
- **Particle semantics:** Transfer population represented by particle
  concentration together with immutable per-particle species mass and charge.
  Whole or fractional source populations may be represented only when a
  deterministic destination slot can preserve their composition. Exact matching
   destination slots may accumulate concentration; otherwise reserve validated
   ascending pre-step free slots. Slots cleared by the same call are not reused.
   Insufficient fixed capacity rejects before the gated commit kernel.
- **Volume semantics:** `particles.volume` remains the sole simulation-volume
  owner. Positive finite prescribed final volumes update in place. Particle
  masses, density, and charge do not scale merely because box volume changes;
  concentrations scale through extensive inventory normalization. Construct all
  edge amounts from pre-step volume, then apply final volume and normalize once.
- **Lifecycle:** Preflight errors leave resident state reusable. A status failure
  before commit leaves particle/gas/volume arrays unchanged, though documented
  scratch may contain plans. Any failure after a commit launch faults the E7-F4
  session; rollback and retry are not promised.
- **Fallback:** E7-F6 resolves capability before setup or after explicit restore.
  Communication never triggers runtime CPU fallback, conversion, `.numpy()`, or
  synchronization during a normal step.

## Security & Compliance

No network, credential, or regulated-data behavior is added. Reject out-of-range
box indices, dynamic callables, malformed map data, cross-device arrays,
forbidden aliases, non-finite/negative transfer values, nonpositive volumes,
outbound overdraw, and capacity exhaustion before commit. Fixed capacities bound
allocations and work. Errors report metadata and reason codes, not array content.
