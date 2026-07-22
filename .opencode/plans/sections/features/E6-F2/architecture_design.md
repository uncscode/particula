# Architecture Design

## High-Level Design

The P2 low-level step receives caller-owned Warp containers and
either a finite nonnegative scalar coefficient or a same-device `wp.float64`
per-box coefficient array. It validates scalar domains and Warp-array metadata.
A scalar is normalized into private active-device storage, while a valid
supplied Warp array is retained by identity. P2 validates launch-safety
concentration metadata and applies the E6-F1 factor `exp(-alpha * time_step)`
only to particle and gas concentrations. Per-box coefficient-value and complete
container-state scans, plus rollback, are P3 work.

```text
E6-F1 coefficient/update contract
             |
caller explicit CPU -> Warp conversion (outside this API)
             |
WarpParticleData + WarpGasData + alpha(scalar | n_boxes) + dt
             |
 P1/P2 scalar/domain + launch-safety metadata preflight
              |
  scalar-zero/zero-time return identical containers (no launch/write)
               |
  P2 device dilution factor per box
        /                         \
particle concentration kernel     gas concentration kernel
        \                         /
same caller-owned containers; all other fields untouched
             |
optional caller explicit Warp -> CPU conversion (outside this API)
```

## Data / API / Workflow Changes

- **Data Model:** No container schema changes. P2 mutates only
  `WarpParticleData.concentration` and `WarpGasData.concentration`; it preserves
  masses, charge, density, volume, molar mass, vapor pressure, partitioning,
  shapes, devices, dtypes, and array identities.
- **API Surface:** P2 exports only
  `particula.gpu.kernels.dilution.dilution_step_gpu(particles, gas,
  coefficient, time_step)`. It accepts a scalar or `(n_boxes,)` same-device
  `wp.float64` coefficient and returns the same particle and gas objects. It is
  exported from `particula.gpu.kernels`, with no private helper exports.
- **Finite Step:** P2 applies E6-F1's `alpha = Q / V` [s^-1] update
  `c_new = c * exp(-alpha * time_step)` in place.
- **Validation Ordering:** P1 rejects invalid scalar coefficient form/domain
  and Warp coefficient dtype/rank before `time_step` or container access.
  After valid `time_step` input, it obtains particle metadata to check per-box
  coefficient shape/device compatibility. Per-box values and complete container
  state remain P3 scope.
- **Workflow Hooks:** E6-F2 depends on E6-F1 fixtures and feeds E6-F9's direct
  GPU process sequence. It introduces no scheduler or high-level runnable.

## Security & Compliance

No network, persistence, permission, or secret changes are involved. Scientific
robustness requires explicit SI units, finite/nonnegative input validation,
atomic failure behavior, and recorded float64 parity tolerances. Process code
must not perform hidden transfers or synchronize for convenience; documentation
must not imply CUDA availability or performance guarantees.
