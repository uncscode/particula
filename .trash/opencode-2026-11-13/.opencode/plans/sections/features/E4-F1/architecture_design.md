# Architecture Design

## High-Level Design

```text
CPU model selection and numeric parameters
  -> fixed-shape Warp thermodynamic configuration (species indexed)
  -> refresh_vapor_pressure_gpu(thermodynamics, gas, temperature)
       -> validate gas buffers, temperature, and configuration
       -> one `(n_boxes, n_species)` Warp launch
       -> overwrite `gas.vapor_pressure`

`condensation_step_gpu(..., thermodynamics=...)`
  -> normalize and validate all entry-point inputs and optional buffers
  -> use normalized `float64` temperature, or device-copy `float32` to `float64`
  -> refresh_vapor_pressure_gpu(thermodynamics, gas, refresh_temperature)
  -> prepare environment properties
  -> transfer and apply condensation mass
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
- **Integration ordering:** `condensation_step_gpu()` completes environment,
  thermodynamics, optional-buffer, and mass-transfer validation before refresh.
  It invokes exactly one refresh after any device-local float32-to-float64 cast
  and before environment-property preparation and mass transfer. Thus failed
  calls leave vapor pressure and particle state untouched.
- **Compatibility:** Existing positional arguments through `mass_transfer`
  remain unchanged; `environment` and the required `thermodynamics` sidecars
  remain keyword-only. Scalar, direct Warp-array, and explicit-environment
  inputs all select the current normalized per-box temperature without host
  formula evaluation or pressure transfer.

## Security & Compliance

No permissions or external I/O are introduced. Robustness requires finite and
bounded numeric inputs, checked shapes/devices, no out-of-bounds species access,
and failure before particle or gas mutation. Tests must cover Warp CPU and skip
CUDA cleanly when unavailable.
