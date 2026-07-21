# Architecture Design

### High-Level Design

E6-F9 is a validation and publication layer over existing direct entry points;
it does not add a process coordinator to production code.

```text
CPU ParticleData/GasData/EnvironmentData fixtures
  -> explicit to_warp_* conversion (once)
  -> caller-owned Warp containers, RNG, scratch, and diagnostics
     -> condensation_step_gpu
     -> coagulation_step_gpu
     -> dilution_step_gpu          [E6-F2]
     -> wall_loss_step_gpu         [E6-F3/E6-F4]
     -> nucleation_step_gpu        [E6-F5/E6-F6/E6-F8]
  -> synchronize at documented checkpoint
  -> explicit from_warp_* restore (once)
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
