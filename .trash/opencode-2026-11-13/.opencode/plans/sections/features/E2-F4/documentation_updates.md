# Documentation Updates

## Required Updates

- `E2-F4-P1` did not land broader docs changes. The phase used
  `particula/gpu/tests/conversion_test.py` as an internal phase-local contract
  record for the current behavior, not as shipped user-facing documentation.
- `E2-F4-P2` updated code-local documentation in
  `particula/gpu/conversion.py` and nearby `WarpGasData` wording to state that
  caller-supplied ordered names are preferred, placeholder names are generated
  only when names are omitted or `None`, and restored `partitioning` must be
  binary `0/1`.
- `E2-F4-P3` updated code-local documentation only:
  - `particula/gpu/conversion.py` now states that
    `to_warp_gas_data(..., vapor_pressure=...)` accepts an optional
    caller-supplied array, requires shape `(n_boxes, n_species)`, and otherwise
    creates a zero-filled GPU buffer with that shape.
  - `particula/gpu/conversion.py` now states that
    `from_warp_gas_data()` is intentionally lossy for `vapor_pressure` and that
    callers must read or save GPU-side values before restore.
  - `particula/gpu/warp_types.py` keeps `vapor_pressure` documented as GPU-only
    helper state.
- `docs/Features/particle-data-migration.md`
  - Added a migration-facing field-authority table for `GasData` versus
    `WarpGasData` covering `name`, `molar_mass`, `concentration`,
    `partitioning`, and `vapor_pressure`.
  - Added explicit round-trip guidance for caller-supplied names,
    placeholder-name fallback, `bool → int32 → bool` `partitioning`, and
    `(n_boxes, n_species)` vapor-pressure sidecar handling.
  - Added a short CPU→GPU→CPU handoff example that preserves ordered names
    outside `WarpGasData` and treats `vapor_pressure` as caller-managed helper
    state.
- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Revised gas-boundary wording from unresolved schema drift to the intentional
    tested contract.
  - Kept remaining limitations explicit: names stay caller-managed,
    `partitioning` stays `bool` on CPU and `int32` on GPU, and
    `vapor_pressure` remains GPU-only helper state with intentionally lossy CPU
    restore.

## Code Documentation

- `#1197` did not update code docstrings, `#1198` updated
  `from_warp_gas_data()` and nearby `WarpGasData` wording, and `#1199` updated
  both gas conversion helpers plus `WarpGasData` vapor-pressure wording.
- `#1200` added a narrow consistency pass in `particula/gas/gas_data.py` so the
  module/class docstrings explicitly state that `GasData` owns only CPU gas
  fields, `vapor_pressure` lives on `WarpGasData`, and semantic name restore
  still depends on caller-supplied ordering.
- Keep `to_warp_gas_data()` and `from_warp_gas_data()` docstrings aligned with
  the shipped name, `partitioning`, and vapor-pressure semantics.
- Keep examples short and consistent with tests; do not introduce runtime
  behavior changes for docstring churn.

## Migration Guidance to Include

- Users with name-keyed logic should pass explicit names or preserve metadata
  sidecars across GPU transfers.
- Users running GPU condensation should compute and pass vapor pressure with
  shape `(n_boxes, n_species)`.
- CPU `GasData` should not be assumed to reconstruct GPU-only vapor-pressure
  buffers; callers who need those values after GPU work must preserve a sidecar
  or read `gpu_data.vapor_pressure` before restore.
- Treat the migration guide authority table as the user-facing summary and the
  roadmap wording as the deeper implementation-facing contract record.
