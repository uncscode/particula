# Architecture Design

## High-Level Design

RNG is session-owned mutable process state, not a scheduler seed argument. A
stable `StreamKey(schema, process_id, logical_box_id)` separates processes and
boxes. Setup derives each initial state from a validated root seed and key;
subsequent calls pass the same buffer with initialization disabled. Storage
permutations move states with logical IDs rather than changing their identity.

```text
root seed + stable logical box IDs + stochastic process IDs
                         |
                  StreamRegistry.setup
                  /                  \
       coagulation/Brownian       wall-loss
       uint32 state by box        uint32 state by box
                  \                  /
           E7-F4 ResidentSession resources
                         |
        E7-F5 resolved enabled process/box execution
                         |
       direct kernels mutate selected stream state in place
                         |
       checkpoint: sync once + stream manifest/state snapshot
                         |
       restart: validate + upload into fresh session -> continue
```

## Data / API / Workflow Changes

- **Data Model:** Add immutable `StreamKey`, `StreamDescriptor`, and checkpoint
  records plus a session-owned `StreamRegistry`. The registry maps stable logical
  box IDs to storage lanes and process-specific same-device `wp.uint32` arrays.
  Root seed, derivation/schema version, process ID, ordered logical IDs, and
  current state are checkpointed. Particle/gas/environment schemas do not change.
- **Seed derivation:** Use a specified deterministic integer mixing algorithm,
  not Python `hash()`, over unsigned root seed, schema/process namespace, and
  logical box ID. Freeze test vectors. Detect duplicate logical IDs and reject
  unsupported identifier forms before allocation or mutation.
- **API Surface:** Session setup accepts logical box IDs and root seed. Explicit
  `initialize_streams()`/`reset_streams()` operations are legal only at lifecycle
  boundaries and can target declared processes/boxes. Normal `step()` never
  reseeds. Inspection returns immutable metadata at setup/checkpoint boundaries,
  not a hidden per-step `.numpy()` readback.
- **Workflow Hooks:** E7-F3 acquires the coagulation resource view. A wall-loss
  adapter uses a separate namespace. E7-F5 passes the resolved enablement mask or
  equivalent validated execution selection so skipped boxes do not consume RNG.
  E7-F4 checkpoint/restart persists and restores the registry as required state.
- **Restart guarantee:** On the same supported backend, device class, process
  configuration, dimensions, logical IDs, and schema version, continuing from a
  checkpoint produces exactly the same mutable RNG states and stochastic outputs
  as an uninterrupted run. Cross-backend/device exactness is not promised.
- **Failure semantics:** All metadata/schema/device checks precede reset or
  process launch. Pre-launch rejection preserves streams. Post-launch failure may
  leave streams and simulation arrays partially advanced, faults the session,
  and cannot be checkpointed as a valid restart state.

## Security & Compliance

This is a simulation RNG, not a cryptographic primitive. Reject duplicate or
unbounded IDs, malformed checkpoint versions, unknown process namespaces,
cross-device arrays, shape/dtype mismatches, and arbitrary deserialized objects.
Resource sizes remain bounded by validated box count. No network, credentials,
silent fallback, executable checkpoint payload, or dynamic import is introduced.
