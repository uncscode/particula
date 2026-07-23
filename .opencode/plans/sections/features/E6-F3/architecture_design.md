# Architecture Design

## High-Level Design

### Shipped P1 transport ownership

P1 establishes `particula.gpu.properties` as the one-way canonical owner for
neutral fp64 scalar transport. `particle_properties.py` owns the migrated
particle radius, Cunningham slip, diffusion, effective-density, and settling
helpers together with device-only `debye_1_wp` and `x_coth_x_wp`; the package
`__init__.py` provides their supported property import surface. Gas viscosity
and mean-free-path remain in `gas_properties.py`. `particula.gpu.dynamics` has
no compatibility definitions or re-exports for moved helpers, so kernels,
dynamics modules, support code, validation, tests, and benchmark collection
import from properties directly.

`cunningham_slip_correction_wp` returns `1.0` for exact zero, evaluates the
standard expression for finite positive Knudsen values, and returns `0.0` for
negative or non-finite values. The two new geometry factors are pure
device-only functions with explicit zero limits and invalid-domain sentinels.
They are foundations for later coefficient work; P1 adds neither coefficient
assembly nor a wall-loss entry point.

### Shipped P2 coefficient assembly

`particula.gpu.dynamics.wall_loss_funcs` now owns two concrete internal
device helpers: `spherical_wall_loss_coefficient_wp` and
`rectangle_wall_loss_coefficient_wp`. Each derives viscosity, mean free path,
Knudsen number, slip, mobility, diffusion, and transport-input settling once,
then assembles the corresponding neutral Crump-Seinfeld coefficient in fp64
and SI `s^-1`. The rectangular helper uses `x_coth_x_wp` to retain the CPU
terms while avoiding direct small-argument coth cancellation.

These pure per-lane helpers do not allocate, transfer, mutate, validate public
inputs, expose a package API, or own state. Configuration and public preflight
are supplied by P3; particle-state mutation and RNG lifecycle remain deferred
to P4-P5. Charged terms remain E6-F4 work.

### Shipped P3 input boundary

`particula.gpu.kernels.wall_loss` owns the frozen host-side
`NeutralWallLossConfig` and `wall_loss_step_gpu` boundary. The entry point is
lazily exported by `particula.gpu.kernels`; its configuration stays a
concrete-module-only API. It validates canonical neutral particle-resolved
configuration, particle schemas/domains, time, environment, and optional RNG
metadata in that order, then returns the identical particle container without
writes. Validation may use private read-only scan storage but does not call
coefficient helpers, allocate output/RNG resources, initialize or advance RNG,
or mutate particle fields.

The feature is a low-level, particle-resolved Warp operation. P3 freezes the
input contract; P4-P5 will add coefficient execution, survival, removal, and
RNG lifecycle behind that unchanged signature.

```text
CPU Spherical/RectangularWallLossStrategy coefficient oracle
                            |
caller-owned WarpParticleData + environment + geometry config + dt
                            |
       complete host/device preflight; failure => no writes/RNG advance
                            |
     P3 successful preflight -> same particle object, no writes or RNG changes
                            |
              P4-P5 deferred: coefficient/removal/RNG execution
```

P3 invalid calls are rejected before mutable-runtime work and preserve supplied
particle and RNG-sidecar state. Execution-time mutation and rollback semantics
remain deferred.

## Data / API / Workflow Changes

- **Data Model:** No container schema changes. `WarpParticleData.masses`,
  `concentration`, and `charge` may be cleared for removed slots. `density`,
  `volume`, all shapes/devices/dtypes, array identities, and survivor state are
  preserved. RNG remains a caller-owned sidecar, not a container field.
- **Configuration:** Add immutable concrete-module configuration describing
  `geometry`, positive finite `wall_eddy_diffusivity`, and either positive
  finite `chamber_radius` or three positive finite `chamber_dimensions`.
  Reject mixed, missing, unknown, or charged terms at capability preflight.
- **API Surface:** Add `wall_loss_step_gpu(particles, temperature, pressure,
  time_step, *, config, rng_seed=0, rng_states=None, initialize_rng=False,
  environment=None)` with final naming frozen in P3. Export the step lazily from
  `particula.gpu.kernels`; keep configuration and primitive helpers in concrete
  modules. Return the same particle object; supplied RNG state is retained by
  identity and is not hidden in the return contract.
- **Environment Inputs:** Follow existing direct-kernel rules: finite positive
  scalar values, active-device `(n_boxes,)` `wp.float64` arrays, hybrid direct
  forms, or explicit `WarpEnvironmentData`; mixing direct values with
  `environment=` fails before mutation. Scalars may use private device buffers;
  supplied arrays are not copied or replaced.
- **RNG Workflow:** P3 validates `rng_seed`, `initialize_rng`, and an optional
  `(n_boxes,)` `wp.uint32` sidecar only. Omitted-sidecar allocation, seeding,
  reset, and advancement are deferred to P5.
- **Epic Integration:** E6-F3 is an independent upstream track and provides the
  neutral coefficient/removal foundation for E6-F4. E6-F9 consumes the direct
  step in integration validation; no scheduler or high-level runnable is added.

## Security & Compliance

No network, permission, persistence, or secret changes are involved. Scientific
robustness requires SI units, finite/positive physical inputs, deterministic
fp64 coefficient parity, statistically justified stochastic bounds, and
validation before caller mutation. Documentation must not claim exact cross-
backend RNG sequences, mandatory CUDA, charged support, hidden transfer,
general multi-box transport, graph capture, or performance guarantees.
