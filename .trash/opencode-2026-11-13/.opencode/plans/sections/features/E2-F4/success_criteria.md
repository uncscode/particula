# Success Criteria

## Functional Criteria

- Gas CPU/GPU field ownership is explicit in code docs and user migration docs.
- `from_warp_gas_data()` preserves supplied names, and omitted names follow one
  tested, documented contract rather than implicit placeholder behavior.
- Wrong-length or empty provided name lists fail explicitly instead of
  degrading into silent placeholder recovery.
- `partitioning` conversion between CPU `bool` and GPU `int32` is explicit and
  stable across round trips, with non-binary restore values rejected before
  bool coercion.
- `to_warp_gas_data()` vapor-pressure behavior is explicit for supplied,
  missing, and invalid-shape inputs.
- `from_warp_gas_data()` vapor-pressure behavior is explicit: the data is either
  preserved through the chosen mechanism or intentionally discarded with no
  silent ambiguity.
- Migration-facing docs now publish the same tested contract and no longer frame
  the gas split as unresolved schema drift.

## Verification Criteria

- `E2-F4-P1` through `E2-F4-P3` now have fast CPU/Warp semantic tests covering
  names supplied, names omitted, explicit `name=None`, invalid name lengths,
  empty name lists, valid binary `partitioning` restore, invalid non-binary
  `partitioning` failures, supplied vapor pressure, missing vapor pressure,
  invalid vapor-pressure shapes, retry-safe correction paths, and GPU-only
  `vapor_pressure` loss on restore.
- `E2-F4-P4` reuses that existing regression contract rather than adding runtime
  behavior, and keeps documentation wording aligned to those tests.
- No CUDA-only requirement is introduced for the semantic test path.
- Focused conversion tests and the changed fast suite pass without requiring
  downstream kernel migration work.

## Documentation Criteria

- `E2-F4-P2` and `E2-F4-P3` made the restore and vapor-pressure transfer
  contracts explicit in code docstrings, and `E2-F4-P4` published the same
  contract in migration and roadmap docs.
- Migration docs state the authoritative name contract, the vapor-pressure
  ownership decision, and any intentional CPU/GPU round-trip loss boundaries.
- Docs/examples do not imply hidden metadata preservation when a sidecar or
  caller-supplied argument is required.
- Docstring consistency updates stay limited to wording and do not imply runtime
  behavior changes.

## Done Signal

Issue `#1172` feature `E2-F4` is complete now that gas round-trip semantics are
explicit, test-backed, and documented closely enough that E2-F5 and E2-F9 can
cite one authoritative contract.
