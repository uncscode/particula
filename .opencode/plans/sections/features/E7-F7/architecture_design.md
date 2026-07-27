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

- **Data model:** Add immutable communication configuration with fixed edge
  capacity, source/destination `int32` arrays, enabled mask, finite nonnegative
  transfer weights or prescribed amounts, transport mode, optional boundary
  accounting, and prescribed final-volume/rate inputs. Shapes and edge capacity
  remain stable for the session.
- **Resident resources:** Add same-device `float64` amount ledgers for gas,
  particle concentration and species inventory, `int32` slot plans/status, and
  documented diagnostics. E7-F4 allocates or validates them once and checkpoints
  mutable state needed for restart.
- **API surface:** Expose only high-level immutable declarations and scheduler
  node configuration through `particula.execution`. Keep concrete Warp kernels,
  scratch records, status buffers, and slot planners module-local.
- **Gas semantics:** Treat `gas.concentration` as mass per volume. Stage
  `concentration * old_volume`, transfer synchronously, account for open
  boundaries explicitly, and divide by validated final volume.
- **Particle semantics:** Transfer population represented by particle
  concentration together with immutable per-particle species mass and charge.
  Whole or fractional source populations may be represented only when a
  deterministic destination slot can preserve their composition. Exact matching
  destination slots may accumulate concentration; otherwise use validated free
  slots. Insufficient fixed capacity rejects before the commit kernel.
- **Volume semantics:** `particles.volume` remains the sole simulation-volume
  owner. Positive finite prescribed final volumes update in place. Particle
  masses, density, and charge do not scale merely because box volume changes;
  concentrations scale through extensive inventory normalization.
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
