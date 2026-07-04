# Dependencies

## Required Dependencies

- `E2-F1`: schema foundation for data-oriented gas and particle containers.
  Implementation should confirm the final `GasData`, `WarpGasData`, and
  conversion-helper baseline before changing semantics.

## Related Sibling Tracks

- `E2-F2`: environment containers may affect where temperature-dependent vapor
  pressure belongs.
- `E2-F3`: environment transfer-helper conventions constrain how any
  vapor-pressure side data reaches GPU code once ownership is decided here.
- Later GPU kernel migration tracks may consume the explicit contract produced
  here.

## Sequencing Constraints

- `E2-F4-P1` can begin after `E2-F1` because it audits current gas and Warp
  behavior rather than finalizing new ownership.
- `E2-F4-P2` can harden name and partitioning semantics from the existing
  baseline, but `E2-F4-P3` should not lock vapor-pressure behavior until the
  `E2-F2`/`E2-F3` gas-versus-environment boundary is documented.
- Documentation in `E2-F4-P4` should publish only the vapor-pressure contract
  proven by `P3`, so E2-F5 and E2-F9 inherit one consistent transfer story.

## Code Dependencies

- `numpy` and `numpy.typing` for CPU arrays and assertions.
- Optional `warp` dependency for GPU structs and tests.
- Existing `pytest.importorskip("warp")` pattern for GPU tests.

## Documentation Dependencies

- `docs/Features/particle-data-migration.md`.
- `docs/Features/Roadmap/data-oriented-gpu.md`.

## External Dependencies

No new external package dependencies are expected.
