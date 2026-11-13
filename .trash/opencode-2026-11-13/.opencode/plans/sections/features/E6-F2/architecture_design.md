# Architecture Design

## High-Level Design

The low-level step receives caller-owned Warp containers and either a finite
nonnegative scalar coefficient or a same-device `wp.float64` per-box
coefficient array. P3 completes read-only preflight in a fixed order:
coefficient form/domain, time, mass schema, per-box coefficient schema/values,
particle concentration schema/values, then gas concentration schema/values.
Masses establish the active device and dimensions; scalar normalization into
private device storage occurs only after all checks pass, while a valid supplied
Warp array is retained by identity. Only then may P2's kernels apply E6-F1's
`exp(-alpha * time_step)` to particle and gas concentrations. Rollback after a
successfully launched-kernel failure remains deferred.

```text
E6-F1 coefficient/update contract
             |
caller explicit CPU -> Warp conversion (outside this API)
             |
WarpParticleData + WarpGasData + alpha(scalar | n_boxes) + dt
             |
 P3 complete ordered read-only preflight
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
- **Validation Ordering:** P3 rejects coefficient form/domain before time or
  container access; time before masses; masses before later fields; per-box
  coefficient schema/values before concentrations; and particle schema/values
  before gas schema/values. All accepted array fields have exact float64,
  same-device Warp schemas and finite nonnegative values where physical.
- **Workflow Hooks:** E6-F2 depends on E6-F1 fixtures and feeds E6-F9's direct
  GPU process sequence. It introduces no scheduler or high-level runnable.

## Security & Compliance

No network, persistence, permission, or secret changes are involved. Scientific
robustness requires explicit SI units, finite/nonnegative input validation,
atomic failure behavior, and recorded float64 parity tolerances. Process code
must not perform hidden transfers or synchronize for convenience; documentation
must not imply CUDA availability or performance guarantees.
