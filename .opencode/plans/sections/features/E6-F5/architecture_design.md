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
  `particula.gpu.kernels.slot_management`. Expose only the intended step helper
  through lazy package exports; keep kernels and scratch helpers private.
- **Mutation Contract:** Activation mutates only selected slot mass,
  concentration, and charge cells. Density, volume, requests, unselected slots,
  shapes, dtypes, devices, array objects, and container identity are unchanged.
- **Workflow Hooks:** E6-F6 uses discovery before choosing exhaustion policy;
  E6-F7/E6-F8 prepare source records and invoke activation only after inventory
  finalization; E6-F9 verifies the integrated contract.
- **Failure Boundary:** Every host-detectable and device-data error is resolved
  before caller diagnostics are cleared or any particle write launches. A
  runtime device failure after successful preflight has no rollback guarantee.

## Security & Compliance

There are no network, permission, or persistence changes. Scientific safety
requires finite nonnegative masses/concentrations, finite charge, strict state
classification, deterministic index order, integer-exact diagnostics, and
failure-before-mutation tests. Documentation must not imply dynamic capacity,
automatic exhaustion recovery, hidden transfer, or graph-capture evidence.
