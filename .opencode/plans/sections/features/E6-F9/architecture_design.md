# Architecture Design

### High-Level Design

E6-F9 is a validation and publication layer over existing direct entry points;
it does not add a process coordinator to production code.

The shipped P1 layer is NumPy-first and test-local. Its deterministic fp64
fixtures, snapshot/ownership helpers, direct inventory/dilution/wall-loss/slot
expectations, and CPU exhaustion-planner assertions establish independent
oracles before P2 composes any direct kernel calls. Its optional Warp mirror
test only verifies fresh container/sidecar schema, identity, and unchanged
values; it does not invoke a process step.

P2 extends that module with a private resident path. It derives all-enabled
partitioning variants from the P1 fixtures, constructs fixed-shape Warp state
and caller-owned sidecars directly, and retains those objects across all five
existing entry points. Intermediate accounting uses synchronized raw device
snapshots; guarded `from_warp_*` conversion is permitted only for final
inspection. This validates composition without creating a production workflow.

```text
test-local all-enabled fixture
  -> caller-owned Warp containers, RNG, scratch, and diagnostics
     -> condensation_step_gpu
     -> coagulation_step_gpu
     -> dilution_step_gpu          [E6-F2]
     -> wall_loss_step_gpu         [E6-F3/E6-F4]
     -> nucleation_step_gpu        [E6-F5/E6-F6/E6-F8]
  -> synchronized raw-device checkpoints
  -> guarded explicit from_warp_* final inspection
  -> process-specific parity, accounting, and diagnostics assertions
```

The order is a fixed example/test scenario, not a general scheduler contract.
Tests take snapshots at device-resident boundaries where needed, but do not
round-trip process state through the host between calls.

### Data / API / Workflow Changes

- **Data Model:** No production container fields or shapes change. Existing
  fixed-shape `WarpParticleData`, `WarpGasData`, `WarpEnvironmentData`, RNG,
  scratch, request, policy, and diagnostic sidecars are reused by identity.
- **API Surface:** No new production API is required. A public example invokes
  the existing direct kernel entry points and public transfer helpers.
- **Accounting:** Condensation and nucleation assert per-box/species
  particle-plus-gas conservation; coagulation asserts mass/charge conservation;
  dilution and wall loss assert their independently calculated removal budgets.
- **Workflow Hooks:** The final phase updates roadmap inventories and E6 plan
  evidence only after all E6-F1 through E6-F8 dependencies are shipped.

### Security & Compliance

No credentials, network calls, or new permissions are introduced. Public inputs
continue to validate before mutation, examples allocate bounded fixed-shape
arrays, and subprocess tests use repository-owned paths. Documentation must not
imply automatic fallback, backend selection, scheduler ownership, CUDA
availability, or production performance evidence.
