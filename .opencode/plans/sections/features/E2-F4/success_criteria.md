# Success Criteria

## Functional Criteria

- Gas CPU/GPU field ownership is explicit in code docs and user migration docs.
- `from_warp_gas_data()` preserves supplied names, and omitted names follow one
  tested, documented contract rather than implicit placeholder behavior.
- `partitioning` conversion between CPU `bool` and GPU `int32` is explicit and
  stable across round trips.
- `to_warp_gas_data()` vapor-pressure behavior is explicit for supplied,
  missing, and invalid-shape inputs.
- `from_warp_gas_data()` vapor-pressure behavior is explicit: the data is either
  preserved through the chosen mechanism or intentionally discarded with no
  silent ambiguity.

## Verification Criteria

- `E2-F4-P1` now has fast CPU/Warp semantic tests covering names supplied,
  names omitted, invalid name lengths, `partitioning` round trips, supplied
  vapor pressure, missing vapor pressure, invalid vapor-pressure shapes, and
  GPU-only `vapor_pressure` loss on restore.
- No CUDA-only requirement is introduced for the semantic test path.
- Focused conversion tests and the changed fast suite pass without requiring
  downstream kernel migration work.

## Documentation Criteria

- In `E2-F4-P1`, regression tests act as the shipped contract documentation.
- Migration docs state the authoritative name contract, the vapor-pressure
  ownership decision, and any intentional CPU/GPU round-trip loss boundaries.
- Docs/examples do not imply hidden metadata preservation when a sidecar or
  caller-supplied argument is required.

## Done Signal

Issue `#1172` feature `E2-F4` is complete when gas round-trip semantics are explicit,
test-backed, and documented closely enough that E2-F5 and E2-F9 can cite one
authoritative contract.
