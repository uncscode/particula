# Dependencies

## Internal Dependencies

- Parent epic E2 must remain the organizing source for Issue #1172 tracks.
- Existing container code must be readable and stable enough for inventory:
  - `particula/particles/particle_data.py`
  - `particula/gas/gas_data.py`
  - `particula/gpu/warp_types.py`
  - `particula/gpu/conversion.py`
- Existing tests provide evidence for current behavior but should not be treated
  as a complete specification until the decision record is published.

## Downstream Dependencies Created by This Feature

- E2-F2 depends on E2-F1 for the future `EnvironmentData` field list, shapes, and
  ownership rules.
- E2-F3 depends on E2-F1 for `WarpEnvironmentData` mirroring and conversion
  semantics.
- E2-F4 depends on E2-F1 for gas versus Warp gas ownership, especially names,
  partitioning, and vapor pressure.
- E2-F5 depends on E2-F1 for scalar-to-per-box compatibility rules.
- E2-F6 through E2-F8 depend on E2-F1 for dtype and support-boundary language.
- E2-F9 depends on E2-F1 for final user-facing data-model documentation.

## External Dependencies

- No new package dependency is expected.
- Warp remains optional and should continue to be guarded by existing
  `WARP_AVAILABLE` patterns.

## Sequencing Notes

This feature should ship early in E2. Downstream tracks may begin research in
parallel, but implementation that adds or changes containers should wait for the
field ownership and shape convention decisions.
