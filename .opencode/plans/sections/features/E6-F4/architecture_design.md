# Architecture Design

## High-Level Design

E6-F4 is an additive charged mode inside the E6-F3 direct wall-loss operation,
not a parallel entry point. P1 shipped host-side immutable configuration and
staged preflight only. The concrete `NeutralWallLossConfig` selects `"neutral"`
or `"charged"`; potential is a signed finite scalar, spherical charged fields
are signed finite scalars, and rectangular charged fields are caller-owned,
same-device `wp.float64` vectors of shape `(3,)`. No execution kernel reads
these values in P1.

```text
E6-F3 config + charged config + WarpParticleData.charge + environment + dt
        |
config validation -> particle schema/device -> rectangular field scan
        -> particle value scans -> time/environment/RNG
        | invalid --> no particle, field, or RNG mutation
        |
active fixed slot --> unchanged E6-F3 neutral coefficient k_neutral
        |
survive with p = exp(-k_neutral * dt) using E6-F3 caller-owned RNG
        +-- survive: preserve all state
        +-- remove: clear all species mass, concentration, and charge
```

The direct P1/P2 wall-loss kernel does not branch on charge or pass
potential/field to device kernels. Thus all slots, including zero-charge slots,
retain the exact E6-F3 neutral coefficient and stochastic path. P2 privately
supplies fp64 device helpers for the self-pair Coulomb ratio (lower-clipped at
`-200`) and image enhancement (absolute ratio, exponent-clipped to `[-50, 50]`,
with `_image_charge_enhancement_wp` returning exact `1.0` for zero charge).
They accept radius, charge, temperature, and explicit physical constants; they
perform no public validation and have no direct-step call site. P3 additionally
privately supplies geometry-scale selection; signed spherical and norm-based
rectangular field resolution with conditional potential scaling; signed
mobility drift with radius/scale guards; and `neutral * factor + drift`
composition sanitized to finite nonnegative fp64 values. P4 alone must connect
these helpers to fixed-slot execution.

## Data / API / Workflow Changes

- **Data Model:** No schema change. `WarpParticleData.charge` remains caller-
  owned `wp.float64` elementary-charge counts shaped like concentration. P1
  validates it and the existing removal path clears it when a slot is removed;
  P1 does not read it for coefficients. Density,
  volume, shapes, devices, dtypes, object identity, and survivor array values
  remain stable. RNG remains a `(n_boxes,)` `wp.uint32` sidecar.
- **Configuration:** P1 extends E6-F3's concrete-module immutable config with
  appended `mode`, `wall_potential` [V], and `wall_electric_field` [V/m]
  fields. The rectangular field is never copied, replaced, or mutated.
- **API Surface:** Reuse E6-F3's `wall_loss_step_gpu(...)` signature and lazy
  step export. Keep configuration and device helpers concrete-module-only.
  Charged mode is selected through configuration; no second public step,
  container conversion, high-level runnable, or backend selector is introduced.
- **Workflow Hooks:** E6-F3 must ship first. E6-F4 reuses its preflight,
  environment forms, fixed-slot mutation, and initialize-once RNG lifecycle.
  E6-F9 later exercises this mode in the integrated explicit-transfer sequence.
- **Failure Boundary:** Malformed configuration, nonfinite rectangular field or
  charge, unsupported
  distributions, inconsistent boxes, wrong shapes/dtypes/devices, and invalid
  environment/time values fail before output allocation, RNG setup, launch, or
  mutation. Runtime launch failure after successful preflight has no rollback.

## Security & Compliance

No network, permission, persistence, or secret changes occur. Scientific
robustness requires SI units, fp64 device math, explicit CPU-oracle equations,
finite input validation, and recorded tolerances. Tests and docs must not imply
exact CPU/Warp RNG streams, hidden fallback, mandatory CUDA, broader charging
physics, dynamic slots, graph capture, or performance evidence.
