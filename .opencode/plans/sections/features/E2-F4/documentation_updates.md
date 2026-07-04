# Documentation Updates

## Required Updates

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

- Update `GasData` and `WarpGasData` docstrings if field authority wording is
  stale.
- Update `to_warp_gas_data()` and `from_warp_gas_data()` docstrings with exact
  name and vapor-pressure semantics.
- Keep examples short and consistent with tests.

## Migration Guidance to Include

- Users with name-keyed logic should pass explicit names or preserve metadata
  sidecars across GPU transfers.
- Users running GPU condensation should compute and pass vapor pressure with
  shape `(n_boxes, n_species)`.
- CPU `GasData` should not be assumed to reconstruct GPU-only vapor-pressure
  buffers unless the final implementation explicitly adds that capability.
