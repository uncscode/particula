# Architecture Design

## High-Level Design

The feature is a low-level, particle-resolved Warp operation. Immutable host
configuration selects one neutral geometry. The public step validates the
configuration, particle schema and physical state, direct/environment inputs,
time step, and optional RNG sidecar before allocation, initialization, or
particle writes. Device functions derive radius and effective density from
fixed particle masses and species density, calculate the CPU-reference
transport terms and coefficient, then draw one survival decision per active
slot. A removal pass clears every mutable per-slot field.

```text
CPU Spherical/RectangularWallLossStrategy coefficient oracle
                            |
caller-owned WarpParticleData + environment + geometry config + dt
                            |
       complete host/device preflight; failure => no writes/RNG advance
                            |
       active slot: concentration > 0 and total species mass > 0
                            |
 radius/density -> transport primitives -> neutral coefficient k [1/s]
                            |
 caller-owned RNG state -> survive with p = exp(-k * dt)
                      /                 \
                  survives             removed
                state unchanged   zero masses, concentration, charge
                      \                 /
          same fixed-shape particle object + advanced RNG sidecar
```

Inactive slots are never sampled or reactivated. Successful calls may advance
RNG and mutate removed slots asynchronously; runtime launch failure has no
rollback guarantee. All contractually detectable invalid input is rejected
before that boundary.

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
- **RNG Workflow:** Omitted RNG uses a seeded call-local convenience buffer.
  Supplied `(n_boxes,)` `wp.uint32` state is reused as-is by default and reset
  only with `initialize_rng=True`. Invalid calls do not initialize or advance it.
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
