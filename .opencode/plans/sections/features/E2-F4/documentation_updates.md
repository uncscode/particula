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
  - Add a field-authority table for `GasData` versus `WarpGasData`.
  - Document name preservation/loss behavior and required user action.
  - Document partitioning dtype conversion.
  - Document vapor-pressure ownership and transfer behavior.
- `docs/Features/Roadmap/data-oriented-gpu.md`
  - Replace schema-drift risk language with the resolved contract after
    implementation.
  - Keep any remaining limitations explicit for downstream tracks.

## Code Documentation

- `#1197` did not update code docstrings, `#1198` updated
  `from_warp_gas_data()` and nearby `WarpGasData` wording, and `#1199` updated
  both gas conversion helpers plus `WarpGasData` vapor-pressure wording.
- Update `GasData` and `WarpGasData` docstrings further only if later phases
  change field-authority wording again.
- Keep `to_warp_gas_data()` and `from_warp_gas_data()` docstrings aligned with
  the shipped name, `partitioning`, and vapor-pressure semantics.
- Keep examples short and consistent with tests.

## Migration Guidance to Include

- Users with name-keyed logic should pass explicit names or preserve metadata
  sidecars across GPU transfers.
- Users running GPU condensation should compute and pass vapor pressure with
  shape `(n_boxes, n_species)`.
- CPU `GasData` should not be assumed to reconstruct GPU-only vapor-pressure
  buffers; callers who need those values after GPU work must preserve a sidecar
  or read `gpu_data.vapor_pressure` before restore.
