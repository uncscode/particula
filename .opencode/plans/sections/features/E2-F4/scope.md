# Scope

## In Scope

- Audit and document current `GasData` / `WarpGasData` field ownership:
  - `name`: CPU metadata, not stored in Warp structs.
  - `molar_mass`: CPU and GPU numeric state.
  - `concentration`: CPU and GPU numeric state with `(n_boxes, n_species)`
    shape.
  - `partitioning`: CPU `bool` authority with GPU `int32` representation.
  - `vapor_pressure`: GPU condensation input buffer, not currently CPU
    `GasData` state.
- Make name handling during `from_warp_gas_data()` explicit through tests and,
  if chosen by implementation, stricter API behavior, warnings, or documented
  placeholder semantics.
- Make vapor-pressure transfer semantics explicit for `to_warp_gas_data()` and
  `from_warp_gas_data()`.
- Add fast, co-located tests under `particula/gpu/tests/` and
  `particula/gas/tests/` where behavior changes or ownership assertions are
  introduced.
- Update migration and roadmap documentation.

## Implemented in E2-F4-P1 (`#1197`)

- Added test-only coverage in `particula/gpu/tests/conversion_test.py`.
- Locked the current public conversion contract for:
  - exact CPU↔GPU shapes and dtypes for `molar_mass`, `concentration`,
    `partitioning`, and `vapor_pressure`;
  - CPU `bool` ↔ GPU `int32` partitioning conversion;
  - explicit `vapor_pressure` preservation and zero-default behavior when
    omitted;
  - placeholder-name restore behavior and name-length mismatch failures;
  - intentional loss of GPU-only `vapor_pressure` when restoring `GasData`.
- Kept production code and broader documentation unchanged for this phase.

## Implemented in E2-F4-P2 (`#1198`)

- Updated `particula/gpu/conversion.py` to make the GPU→CPU restore contract
  explicit for names and `partitioning`.
- Preserved caller-supplied ordered names when provided and documented
  placeholder-name generation for omitted or `None` names.
- Rejected wrong-length and empty provided name lists with explicit
  actual/expected count messaging.
- Preserved GPU-only `vapor_pressure` drop behavior on restore while tightening
  `_restore_partitioning_bool()` so only binary `0/1` values are accepted
  before CPU bool conversion.
- Extended `particula/gpu/tests/conversion_test.py` with focused coverage for
  the shipped name contract, binary `partitioning` validation, multi-box shape
  preservation, and retry-safe correction paths.

## Out of Scope

- Rewriting GPU condensation kernels beyond any small compatibility updates
  needed for clarified `WarpGasData` semantics.
- Adding a broad thermodynamic state model to `GasData` without alignment from
  `E2-F2` and `E2-F3`.
- Replacing Warp structs with string-capable metadata containers.
- Changing public `GasSpecies` vapor-pressure strategy behavior except as
  needed to clarify conversion boundaries.
- Performance optimization of gas transfer paths beyond preserving current
  shape and dtype expectations.

## Non-Goals

- No standalone testing phase; each implementation phase includes the tests
  required for that phase.
- No silent compatibility break without tests and migration documentation.
