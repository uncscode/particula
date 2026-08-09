# Architecture Design

## High-Level Design

Communication is an explicit, opt-in scheduler operation over authoritative
resident state. A validated map and preallocated scratch produce a synchronous
update: every outgoing amount is computed from pre-node state, accumulated into
ledgers, checked, then committed. This avoids edge-order dependence and in-place
donor/receiver races.

```text
PrescribedCommunication + PrescribedVolumeUpdate
                    |
        read-only capability/schema preflight
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
        apply prescribed positive new volumes
        concentration = final amount / final volume
                    |
        commit once; mark derived gas state stale
                    |
 E7-F5 environment/thermodynamic refresh and process graph
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
- **Data model (later phases):** The fixed edge capacity, source/destination
  `int32` arrays, enabled mask, finite nonnegative inverse-time rates,
  transport mode, and optional final volumes remain stable for the session.
- **Resident resources:** Add same-device `float64` amount ledgers for gas,
  particle concentration and species inventory, `int32` slot plans/status, and
  documented diagnostics. E7-F4 allocates or validates them once and checkpoints
  mutable state needed for restart.
- **API surface:** P1 declarations and validation are direct-import-only from
  `particula.execution.communication`; they are not exported through
  `particula.execution` or the package. Later scheduler-facing configuration
  remains deferred.
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
- **Ordering:** Communication/volume execution is a canonical pre-process
  barrier. If it changes gas concentration or volume, dependent diagnostics and
  thermodynamic state are invalidated. E7-F5's environment update and derived
  vapor-pressure/saturation refresh still precede consuming physics processes.
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
