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
