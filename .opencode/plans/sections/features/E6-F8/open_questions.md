# Open Questions

All E6-F8 planning questions were resolved on 2026-07-21 from the selected
E6-F7 source model and existing concrete direct-kernel conventions.

- [x] Which concrete-module sidecar names are frozen?
  - Decision: use `NucleationConfig`, `NucleationScratchBuffers`,
    `NucleationFinalizedDemandBuffers`, and `NucleationDiagnosticBuffers` under
    `particula.gpu.kernels.nucleation`. Bindings are frozen while contained Warp
    arrays remain mutable and caller-owned. P1 remains unexported; only the
    eventual P5 `nucleation_step_gpu` is a candidate kernel-package export.
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
- [x] How does P3 represent demand beyond available slots?
  - Decision (#1440): retain the full exact, representable int32 provisional
    count in `accepted_counts`; write only the ascending E6-F5 free-slot prefix
    to `selected_slot_indices` and `-1` elsewhere. P4 owns capacity policy and
    activation.
- [x] What is P3's failure and synchronization boundary?
  - Decision (#1440): conversion rejection precedes caller-output writes and
    E6-F5 rejection preserves its diagnostics. A launched E6-F5 or P3 writer
    has no rollback guarantee; callers synchronize before reading outputs.
- [x] How does P4 select and finalize an exhaustion policy?
  - Decision (#1441): preserve P2 accepted demand and P3 staging sidecars as
    immutable handoffs; use separate P4 workspace. Select resampling only when
    it releases the complete deficit, then select optional scaling for remaining
    exhausted rows. Derive final counts from post-policy demand-volume products
    and write final demand/count/free-prefix diagnostics without truncation.
- [x] What is P4's all-box failure boundary?
  - Decision (#1441): malformed/stale handoffs, invalid P4/nested buffers,
    insufficient scratch, impossible policies, and invalid final products reject
    before P4 writes or primitive entry, preserving all caller state. An entered
    E6-F6 primitive retains its documented mutation contract; P4 provides no
    cross-primitive rollback.
