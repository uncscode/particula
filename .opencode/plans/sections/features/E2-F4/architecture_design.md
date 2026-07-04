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

- **Data Model:** No new required field should be added to `GasData` unless the
  final implementation proves CPU ownership of vapor pressure is required and
  compatible with `E2-F1`, `E2-F2`, and `E2-F3`. Preferred design keeps vapor
  pressure as explicit process state, a GPU transfer buffer, or sidecar derived
  from strategies and environment state; do not make it CPU `GasData` or
  `EnvironmentData` ownership.
- **API Surface:** `to_warp_gas_data()` and `from_warp_gas_data()` are the
  expected API points for clarified semantics. Candidate changes include a
  stricter missing-name policy, clearer optional arguments, warnings, or helper
  docstrings. Backward compatibility must be tested and documented.
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
