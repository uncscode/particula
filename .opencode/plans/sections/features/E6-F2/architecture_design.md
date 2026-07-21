# Architecture Design

## High-Level Design

The public low-level step receives caller-owned Warp containers and either a
finite nonnegative scalar coefficient or a same-device per-box coefficient
array. Host-side preflight validates metadata and scalar values; device-side
validation scans device-resident physical state before any mutation. A scalar
may be normalized into private device storage, while a supplied Warp array is
used without replacement. Separate fixed-shape kernels update particle and gas
concentrations with exactly the finite-step factor frozen by E6-F1.

```text
E6-F1 coefficient/update contract
             |
caller explicit CPU -> Warp conversion (outside this API)
             |
WarpParticleData + WarpGasData + alpha(scalar | n_boxes) + dt
             |
complete structural/domain/state preflight -- failure => no writes
             |
device dilution factor per box
        /                         \
particle concentration kernel     gas concentration kernel
        \                         /
same caller-owned containers; all other fields untouched
             |
optional caller explicit Warp -> CPU conversion (outside this API)
```

## Data / API / Workflow Changes

- **Data Model:** No container schema changes. Mutate only
  `WarpParticleData.concentration` and `WarpGasData.concentration`; preserve
  masses, charge, density, volume, molar mass, vapor pressure, partitioning,
  shapes, devices, dtypes, and array identities.
- **API Surface:** Add `dilution_step_gpu(particles, gas, coefficient,
  time_step)` (final keyword-only choices frozen in P1) to
  `particula.gpu.kernels`. Accept scalar or `(n_boxes,)` same-device
  `wp.float64` coefficient input. Return the same particle and gas objects.
- **Finite Step:** Reuse T1's canonical update exactly, including any bounded
  explicit or exponential rule. Do not infer a second integration policy.
- **Validation Ordering:** Validate contract exclusivity/types, dimensions,
  dtypes/devices, physical domains, and finite/nonnegative concentration state
  before allocation or kernel launch. Compute/validate any required factor
  before concentration writes so particle and gas cannot diverge on failure.
- **Workflow Hooks:** E6-F2 depends on E6-F1 fixtures and feeds E6-F9's direct
  GPU process sequence. It introduces no scheduler or high-level runnable.

## Security & Compliance

No network, persistence, permission, or secret changes are involved. Scientific
robustness requires explicit SI units, finite/nonnegative input validation,
atomic failure behavior, and recorded float64 parity tolerances. Process code
must not perform hidden transfers or synchronize for convenience; documentation
must not imply CUDA availability or performance guarantees.
