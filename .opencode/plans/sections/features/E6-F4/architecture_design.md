# Architecture Design

## High-Level Design

E6-F4 is an additive charged mode inside the E6-F3 direct wall-loss operation,
not a parallel entry point. Host-side immutable configuration selects neutral or
charged capability and normalizes geometry-specific scalar/vector fields.
Preflight validates every detectable contract before allocation, RNG
initialization, launch, or particle write. Device code computes E6-F3's neutral
coefficient and conditionally composes the charged terms per active slot.

```text
CPU ChargedWallLossStrategy oracle
        | image factor, resolved field, drift, clipping
        v
E6-F3 config + charged config + WarpParticleData.charge + environment + dt
        |
complete host/device preflight ---- invalid --> no particle/RNG mutation
        |
active fixed slot --> E6-F3 neutral coefficient k_neutral
        |
        +-- charge == 0 --> k = k_neutral (exact fallback)
        |
        +-- charge != 0 --> phi = exp(clip(abs(coulomb_ratio), -50, 50))
                            E = |configured field| + potential / geometry scale
                            drift = signed mobility(charge, radius, T) * E / scale
                            k = finite_nonnegative(k_neutral * phi + drift)
        |
survive with p = exp(-k * dt) using E6-F3 caller-owned RNG
        +-- survive: preserve all state
        +-- remove: clear all species mass, concentration, and charge
```

The zero-charge branch must use the same neutral value and stochastic decision
machinery as E6-F3. A configured field without charge does not alter the
coefficient. Nonzero charge retains image-charge enhancement when wall
potential is zero. Potential-derived field and explicit field magnitude are
additive, while charge sign is preserved by the drift term before final
nonnegative clipping.

## Data / API / Workflow Changes

- **Data Model:** No schema change. `WarpParticleData.charge` remains caller-
  owned `wp.float64` elementary-charge counts shaped like concentration. It is
  read for coefficients and cleared only when a slot is removed. Density,
  volume, shapes, devices, dtypes, object identity, and survivor array values
  remain stable. RNG remains a `(n_boxes,)` `wp.uint32` sidecar.
- **Configuration:** Extend E6-F3's concrete-module immutable config with a
  bounded charged mode, finite `wall_potential` [V], and finite scalar or
  length-three `wall_electric_field` [V/m]. Spherical geometry accepts scalar
  field magnitude; rectangular geometry accepts the CPU-compatible vector form
  resolved by Euclidean norm. Defaults are zero without disabling image charge.
- **API Surface:** Reuse E6-F3's `wall_loss_step_gpu(...)` signature and lazy
  step export. Keep configuration and device helpers concrete-module-only.
  Charged mode is selected through configuration; no second public step,
  container conversion, high-level runnable, or backend selector is introduced.
- **Workflow Hooks:** E6-F3 must ship first. E6-F4 reuses its preflight,
  environment forms, fixed-slot mutation, and initialize-once RNG lifecycle.
  E6-F9 later exercises this mode in the integrated explicit-transfer sequence.
- **Failure Boundary:** Malformed configuration, nonfinite charge, unsupported
  distributions, inconsistent boxes, wrong shapes/dtypes/devices, and invalid
  environment/time values fail before output allocation, RNG setup, launch, or
  mutation. Runtime launch failure after successful preflight has no rollback.

## Security & Compliance

No network, permission, persistence, or secret changes occur. Scientific
robustness requires SI units, fp64 device math, explicit CPU-oracle equations,
finite input validation, and recorded tolerances. Tests and docs must not imply
exact CPU/Warp RNG streams, hidden fallback, mandatory CUDA, broader charging
physics, dynamic slots, graph capture, or performance evidence.
