# Architecture Design

## High-Level Design

```text
CPU model selection and numeric parameters
  -> fixed-shape Warp thermodynamic configuration (species indexed)
  -> condensation_step_gpu(...)
       -> normalize current temperature with _ensure_environment_arrays()
       -> validate configuration against gas species count and active device
       -> launch vapor-pressure refresh
            temperature[n_boxes] + mode/parameters[n_species]
            -> gas.vapor_pressure[n_boxes, n_species]
       -> launch existing condensation mass-transfer kernel
```

The configuration is process configuration, not gas state, and should use a
dedicated typed container or explicit keyword-only arrays. Mode values are
compact `int32`; parameters and output are `float64`. Species position is the
identity mapping and must align with gas molar-mass ordering.

## Data / API / Workflow Changes

- **Data model:** Add numeric model mode and fixed-shape parameter storage. Do
  not extend CPU `GasData`; retain `WarpGasData.vapor_pressure` as derived,
  mutable helper state.
- **API surface:** Extend `condensation_step_gpu()` with keyword-only
  thermodynamic configuration to preserve existing positional calls. Raw Warp
  helpers remain module-internal unless separately approved.
- **Formula dispatch:** Constant mode reads a nonnegative pressure parameter.
  Buck mode converts Kelvin to Celsius and applies the CPU reference's ice/water
  piecewise expression at 273.15 K.
- **Mutation order:** Complete host-side metadata validation before launch;
  refresh pressure before mass transfer. E4-F3 will invoke this primitive before
  each future substep.
- **Compatibility:** An omitted configuration must have one documented behavior:
  fail early per issue #1272 or use an explicit legacy/static mode. Silent stale
  state is forbidden.

## Security & Compliance

No permissions or external I/O are introduced. Robustness requires finite and
bounded numeric inputs, checked shapes/devices, no out-of-bounds species access,
and failure before particle or gas mutation. Tests must cover Warp CPU and skip
CUDA cleanly when unavailable.
