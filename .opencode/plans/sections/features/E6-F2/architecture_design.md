# Architecture Design

## High-Level Design

The concrete-only P1 low-level step receives caller-owned Warp containers and
either a finite nonnegative scalar coefficient or a same-device `wp.float64`
per-box coefficient array. It validates scalar domains and Warp-array metadata.
A scalar is normalized into private active-device storage, while a valid
supplied Warp array is retained by identity. P1 launches no kernel and writes
no caller state; per-box coefficient-value and complete container-state scans
are P3 work. P2 will add fixed-shape concentration kernels using the E6-F1
factor `exp(-alpha * time_step)`.

```text
E6-F1 coefficient/update contract
             |
caller explicit CPU -> Warp conversion (outside this API)
             |
WarpParticleData + WarpGasData + alpha(scalar | n_boxes) + dt
             |
 P1 scalar/domain + coefficient metadata preflight -- failure => no writes
              |
 P1 return identical containers (no launch/write)
              |
 P2/P3 complete preflight and device dilution factor per box
        /                         \
particle concentration kernel     gas concentration kernel
        \                         /
same caller-owned containers; all other fields untouched
             |
optional caller explicit Warp -> CPU conversion (outside this API)
```

## Data / API / Workflow Changes

- **Data Model:** No container schema changes. P1 mutates no fields. P2 will
  mutate only `WarpParticleData.concentration` and `WarpGasData.concentration`;
  it will preserve
  masses, charge, density, volume, molar mass, vapor pressure, partitioning,
  shapes, devices, dtypes, and array identities.
- **API Surface:** P1 adds concrete-module-only
  `particula.gpu.kernels.dilution.dilution_step_gpu(particles, gas,
  coefficient, time_step)`. It accepts a scalar or `(n_boxes,)` same-device
  `wp.float64` coefficient and returns the same particle and gas objects.
  P1 deliberately does not re-export it from `particula.gpu.kernels`; P2 owns
  that export.
- **Finite Step:** P1 records E6-F1's `alpha = Q / V` [s^-1] and future P2
  update `c_new = c * exp(-alpha * time_step)` without executing it.
- **Validation Ordering:** P1 rejects invalid coefficient form/domain before
  `time_step` and container access, and invalid `time_step` before container
  access. Per-box values and complete container state remain P3 scope.
- **Workflow Hooks:** E6-F2 depends on E6-F1 fixtures and feeds E6-F9's direct
  GPU process sequence. It introduces no scheduler or high-level runnable.

## Security & Compliance

No network, persistence, permission, or secret changes are involved. Scientific
robustness requires explicit SI units, finite/nonnegative input validation,
atomic failure behavior, and recorded float64 parity tolerances. Process code
must not perform hidden transfers or synchronize for convenience; documentation
must not imply CUDA availability or performance guarantees.
