# Architecture Design

## High-Level Design

`GasData` remains the CPU-side numeric state container. `WarpGasData` remains a
kernel-friendly GPU container that cannot store Python strings and uses Warp
dtypes. This feature tightens the conversion boundary instead of forcing both
containers into an identical schema.

```text
GasSpecies + vapor-pressure strategies
        |
        v
GasData(name, molar_mass, concentration, partitioning: bool)
        | to_warp_gas_data(data, vapor_pressure=...)
        v
WarpGasData(molar_mass, concentration, vapor_pressure, partitioning: int32)
        |
        | from_warp_gas_data(gpu_data, name=...)
        v
GasData(name supplied or explicit fallback, partitioning: bool)
```

## Data / API / Workflow Changes

- **Data Model:** `E2-F4-P1` did not add or remove any fields. The landed work
  treats the existing implementation as authoritative for now: CPU `GasData`
  owns `name`, `molar_mass`, `concentration`, and boolean `partitioning`, while
  `WarpGasData` omits `name`, stores `partitioning` as `int32`, and carries a
  GPU-only `vapor_pressure` buffer.
- **API Surface:** `E2-F4-P1` changed no runtime API behavior. It tightened the
  contract by adding regression tests around `to_warp_gas_data()` and
  `from_warp_gas_data()` rather than introducing stricter missing-name or
  vapor-pressure semantics in production code.
- **Workflow Hooks:** GPU tests continue to use `pytest.importorskip("warp")`
  and `device="cpu"` where possible so that behavior can be validated without
  CUDA-only assumptions.

## Ownership Decisions to Encode

- `name`: authoritative on CPU or external metadata sidecar; absent from
  `WarpGasData` by design. CPU restoration should prefer caller-supplied names
  or external index-map metadata, not embedded name preservation in the Warp
  struct.
- `partitioning`: authoritative as CPU `bool`; represented as GPU `int32` for
  Warp compatibility.
- `vapor_pressure`: operational GPU condensation input with explicit transfer
  behavior; not silently confused with CPU `GasData` ownership.

## Security & Compliance

No new permissions or external services are introduced. The main robustness
requirement is preventing silent data loss from propagating into scientific
results without an explicit test-backed contract.
