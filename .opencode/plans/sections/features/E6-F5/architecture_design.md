# Architecture Design

## High-Level Design

CPU and Warp implementations share a documented truth table rather than an
implicit radius test. Existing slots are classified before any output or
particle write. Discovery produces fixed-shape, deterministic sidecars;
activation consumes the same ordering and only launches after capacity and all
request records are valid.

```text
ParticleData / WarpParticleData + fixed-shape activation requests
                         |
                 complete read-only preflight
        shape/dtype/device/finiteness/state/request/capacity
             | invalid                     | valid
             v                             v
      no particle or sidecar write   classify each fixed slot
                                      | active: conc>0 && sum(mass)>0
                                      | free: conc=0 && all mass=0 && charge=0
                                      v
                         ascending free indices + exact counts
                                      |
                 request rank r -> free index rank r
                                      |
                       in-place mass/concentration/charge writes
                                      |
                      exact activated-count diagnostic
```

Every unrequested free-index cell is `-1`. No compaction occurs. Diagnostics
describe the completed operation exactly: discovery reports pre-operation
active/free counts; activation reports the number written per box and updates
post-operation active/free counts from those validated integers, without a
floating-point reduction.

## Data / API / Workflow Changes

- **Data Model:** No container field changes. CPU diagnostics use NumPy `int32`;
  GPU diagnostics/free indices are caller-owned, same-device `wp.int32` arrays
  shaped `(n_boxes,)` and `(n_boxes, n_particles)`. Supplied buffers are returned
  by identity. Request arrays stay fixed shape; `requested_counts[n]` identifies
  the valid prefix per box.
- **API Surface:** Add CPU inspection/activation helpers under
  `particula.particles.slot_management` and low-level GPU entry points under
  `particula.gpu.kernels.slot_management`. P3 shipped the concrete-module-only
  `get_slot_diagnostics_gpu`; it is deliberately not exported through
  `particula.gpu.kernels` or a package-level API. Keep kernels and scratch
  helpers private.
  P1 delivered the read-only CPU API
  `get_slot_diagnostics(data)`, re-exported from `particula.particles`.
  P2 delivered the CPU-only direct-import
  `activate_slots(data, request_masses, request_concentration, request_charge,
  requested_counts)` API; it is intentionally not re-exported from
  `particula.particles`. P3 returns the supplied same-device `wp.int32`
   `free_indices`, `active_counts`, and `free_counts` sidecars by identity after
   read-only classification of mass, concentration, and charge; density and
   volume are neither read nor validated. P4 ships package-exported
   `activate_slots_gpu`, which returns supplied `activated_counts`,
   `free_indices`, `active_counts`, and `free_counts` sidecars by identity.
   It validates same-device schemas, ownership and aliasing, existing state,
   selected request prefixes, and per-box capacity before mapping each prefix
   rank to an ascending free slot.
- **Mutation Contract:** Activation mutates only selected slot mass,
  concentration, and charge cells. Density, volume, requests, unselected slots,
  shapes, dtypes, devices, array objects, and container identity are unchanged.
- **Workflow Hooks:** E6-F6 uses discovery before choosing exhaustion policy;
  E6-F7/E6-F8 prepare source records and invoke activation only after inventory
  finalization; E6-F9 verifies the integrated contract.
- **Failure Boundary:** Every host-detectable and device-data error is resolved
  before caller diagnostics are cleared or any particle write launches. A
  runtime device failure after successful preflight has no rollback guarantee.
  The shipped P1 discovery path raises exactly
  `ValueError("Invalid particle slot state.")` before allocating or returning
  diagnostics when any slot is neither active nor free. P2 validates destination
  schema and writability, request schema and non-aliasing, selected prefixes,
  and free capacity globally before its first assignment; it then uses P1's
  ascending free-index ordering and returns a fresh per-box `np.int32`
  activated-count array. P3 schema and state rejection occur before a writer
  launch, preserving all supplied diagnostic sidecars. Its single device status
  readback is limited to the private invalid-state flag; successful diagnostics
   remain device-resident. P4 similarly rejects before any caller mutation and
   does not promise rollback after its activation writer launches.

## Security & Compliance

There are no network, permission, or persistence changes. Scientific safety
requires finite nonnegative masses/concentrations, finite charge, strict state
classification, deterministic index order, integer-exact diagnostics, and
failure-before-mutation tests. Documentation must not imply dynamic capacity,
automatic exhaustion recovery, hidden transfer, or graph-capture evidence.
