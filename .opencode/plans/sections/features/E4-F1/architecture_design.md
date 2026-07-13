# Architecture Design

## High-Level Design

```text
CPU model selection and numeric parameters
  -> fixed-shape Warp thermodynamic configuration (species indexed)
  -> refresh_vapor_pressure_gpu(thermodynamics, gas, temperature)
       -> validate gas buffers, temperature, and configuration
       -> one `(n_boxes, n_species)` Warp launch
       -> overwrite `gas.vapor_pressure`

`condensation_step_gpu(...)` retains the P1 sidecar validation boundary;
refresh integration is deferred to P3.
```

The configuration is process configuration, not gas state, and uses a typed,
keyword-only operation sidecar. Mode values are compact `int32`; parameters and
the molar-mass reference are `float64`. Species position is the identity mapping
and must align with gas molar-mass ordering.

## Data / API / Workflow Changes

- **Data model:** `ThermodynamicsConfig` is a frozen Python dataclass holding
  caller-owned Warp buffers; no CPU or Warp container schema changed.
- **API surface:** `refresh_vapor_pressure_gpu` is exported only by the concrete
  `thermodynamics` module, not `particula.gpu.kernels`; its documented import
  path preserves the package's direct-step API boundary.
- **Mode contract:** Concrete-module constants reserve constant parameter zero
  for pressure in Pa. Buck's four parameter slots remain reserved and unused;
  evaluation uses the canonical fixed water/ice equations.
- **Execution:** All type, dtype, shape, device, physical-temperature, and
  configuration validation completes before the sole launch, leaving
  `gas.vapor_pressure` unchanged for invalid calls. Private helpers and the
  kernel remain in `thermodynamics.py` and compute in `float64`.
- **Ordering:** After active-device context is established, metadata checks occur
  before one readback each of sidecar fields and gas molar mass. Validation
  precedes defaults, caller mass-transfer access, allocation, and launch.
- **Compatibility:** The standalone primitive mutates only caller-owned vapor
  pressure. Condensation does not invoke it until P3.

## Security & Compliance

No permissions or external I/O are introduced. Robustness requires finite and
bounded numeric inputs, checked shapes/devices, no out-of-bounds species access,
and failure before particle or gas mutation. Tests must cover Warp CPU and skip
CUDA cleanly when unavailable.
