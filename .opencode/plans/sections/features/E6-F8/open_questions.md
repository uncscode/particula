# Open Questions

All E6-F8 planning questions were resolved on 2026-07-21 from the selected
E6-F7 source model and existing concrete direct-kernel conventions.

- [x] Which concrete-module sidecar names are frozen?
  - Decision: use `NucleationConfig`, `NucleationScratchBuffers`,
    `NucleationFinalizedDemandBuffers`, and `NucleationDiagnosticBuffers` under
    `particula.gpu.kernels.nucleation`. Bindings are frozen while contained Warp
    arrays remain mutable and caller-owned. Re-export only `nucleation_step_gpu`
    through `particula.gpu.kernels`.
- [x] Which environmental inputs are required and in what forms?
  - Decision: pressure is not required. Temperature is required for the model's
    declared validity interval; saturation ratio is required only when its gate
    is configured. Each direct per-box input accepts a Python/NumPy floating
    scalar or same-device `wp.float64 (n_boxes,)` array. An explicit environment
    may supply temperature and `(n_boxes, n_species)` saturation state; mixed
    direct/environment sources and host arrays are rejected.
- [x] Which E6-F6 scratch fields and scale bounds does the adapter consume?
  - Decision: consume E6-F6-owned `ExhaustionScratchBuffers` rather than
    duplicate fields. It carries int32 `sorted_indices`, `retained_indices`, and
    `output_count`, plus float64 `output_weight`, `output_mass`, and
    `output_charge`, with fixed box/slot/species shapes. Raw-count scaling uses
    `0<s<=1` and the caller-configured minimum frozen by E6-F6.
- [x] Which deterministic CPU/GPU tolerances apply?
  - Decision: counts, indices, policy codes, sentinels, and no-op zeros are
    exact. Rates, event counts, scales, finalized masses, and per-box/species
    conservation begin at `rtol=1e-12`, with `atol=1e-30` for mass and gas
    concentration. Any relaxation must be fixture-specific and measured.
- [x] May the direct step resize arrays or fall back to CPU?
  - Decision: no. It must use E6-F6 or fail before mutation.
- [x] Is a high-level GPU runnable part of E6-F8?
  - Decision: no. Orchestration and backend selection remain in Epic G.
