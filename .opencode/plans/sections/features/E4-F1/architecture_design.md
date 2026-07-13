# Architecture Design

## High-Level Design

```text
CPU model selection and numeric parameters
  -> fixed-shape Warp thermodynamic configuration (species indexed)
  -> condensation_step_gpu(...)
       -> normalize current temperature with _ensure_environment_arrays()
        -> validate configuration against gas species count and active device
        -> validate sidecar schema and ordered gas molar-mass fingerprint
        -> continue existing condensation mass-transfer path
```

The configuration is process configuration, not gas state, and uses a typed,
keyword-only operation sidecar. Mode values are compact `int32`; parameters and
the molar-mass reference are `float64`. Species position is the identity mapping
and must align with gas molar-mass ordering.

## Data / API / Workflow Changes

- **Data model:** `ThermodynamicsConfig` is a frozen Python dataclass holding
  caller-owned Warp buffers; no CPU or Warp container schema changed.
- **API surface:** Extend `condensation_step_gpu()` with keyword-only
  thermodynamic configuration to preserve existing positional calls. Raw Warp
  helpers remain module-internal unless separately approved.
- **Mode contract:** Concrete-module constants reserve constant parameter zero
  for pressure in Pa and Buck parameters for reference pressure plus three
  coefficients. P1 validates only schema, supported codes, and finite
  non-negative values; it does not evaluate their semantics.
- **Ordering:** After active-device context is established, metadata checks occur
  before one readback each of sidecar fields and gas molar mass. Validation
  precedes defaults, caller mass-transfer access, allocation, and launch.
- **Compatibility:** Omitted required configuration fails before allocation,
  launch, or mutation. P1 intentionally preserves the existing vapor-pressure
  state rather than selecting a compatibility behavior or refreshing it.

## Security & Compliance

No permissions or external I/O are introduced. Robustness requires finite and
bounded numeric inputs, checked shapes/devices, no out-of-bounds species access,
and failure before particle or gas mutation. Tests must cover Warp CPU and skip
CUDA cleanly when unavailable.
