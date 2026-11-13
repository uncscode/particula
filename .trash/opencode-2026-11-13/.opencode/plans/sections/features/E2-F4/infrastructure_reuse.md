# Infrastructure Reuse

## Existing Code to Reuse

- `particula/gas/gas_data.py`
  - `GasData` dataclass and validation in `__post_init__`.
  - `from_species()` / `to_species()` conversion helpers, especially the
    existing decision that vapor-pressure strategies are behavior supplied
    outside `GasData`.
- `particula/gas/gas_data_builder.py`
  - Builder validation and broadcast patterns for shape-safe multi-box gas
    data.
- `particula/gpu/warp_types.py`
  - `WarpGasData` struct definition and docstring documenting omitted names,
    `int32` partitioning, and GPU vapor-pressure field.
- `particula/gpu/conversion.py`
  - `to_warp_gas_data()` and `from_warp_gas_data()` as the central transfer
    boundary for this feature.
- `particula/gpu/kernels/condensation.py`
  - Existing validation that GPU vapor pressure has shape
    `(n_boxes, n_species)`.

## Existing Tests to Extend

- `particula/gpu/tests/conversion_test.py`
  - CPU-to-Warp gas conversion tests.
  - Warp-to-CPU round-trip and placeholder-name tests.
- `particula/gpu/tests/warp_types_test.py`
  - Warp gas shape and dtype tests.
- `particula/gas/tests/gas_data_test.py`
  - CPU gas schema and facade-conversion tests.
- `particula/gas/tests/gas_data_builder_test.py`
  - Builder shape and validation coverage.

## Documentation to Reuse

- `docs/Features/Roadmap/data-oriented-gpu.md` already calls out schema drift
  and should become the concise roadmap/status reference.
- `docs/Features/particle-data-migration.md` already documents migration
  ownership and should receive the detailed gas round-trip semantics.
