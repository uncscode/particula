# Overview

## Problem Statement

Particula has mature CPU wall-loss strategies for neutral spherical and
rectangular chambers, but no low-level Warp implementation. A fixed-shape GPU
timestep therefore cannot apply neutral particle-resolved wall loss without a
host transfer or unsupported fallback. The missing path must preserve the CPU
coefficient equations and stochastic survival semantics while safely clearing
removed fixed slots and advancing caller-owned RNG state across timesteps.

## Value Proposition

E6-F3 supplies a parity-tested direct GPU wall-loss step for the two neutral
geometries. It keeps particle arrays resident and allocation-stable, makes RNG
ownership explicit, rejects invalid calls before mutation, and establishes the
neutral foundation required by sibling E6-F4's charged wall-loss extension.

## Delivered Phase: E6-F3-P1 (#1401)

P1 shipped the reusable neutral fp64 Warp transport foundation, not a wall-loss
step. `particula.gpu.properties` is now the sole owner and import surface for
the migrated particle radius, diffusion, effective-density, settling, and slip
helpers; legacy definitions and re-exports under `particula.gpu.dynamics` were
removed and consumers were migrated. The phase also adds device-only
`debye_1_wp` and `x_coth_x_wp` geometry factors and defines safe zero/invalid
behavior for `cunningham_slip_correction_wp`.

No coefficient assembly, wall-loss configuration or API, slot removal/RNG
behavior, charged physics, or CPU behavior shipped in P1. Those remain the
scope of later E6-F3 phases and E6-F4.

## Delivered Phase: E6-F3-P2 (#1402)

P2 adds the concrete, internal fp64 Warp helpers
`spherical_wall_loss_coefficient_wp` and
`rectangle_wall_loss_coefficient_wp` in
`particula/gpu/dynamics/wall_loss_funcs.py`. They compose the P1 transport and
geometry primitives to implement the neutral Crump-Seinfeld spherical and
rectangular coefficients in SI units (`s^-1`), including the cancellation-safe
rectangular `x_coth_x_wp` form.

`particula/gpu/dynamics/tests/wall_loss_funcs_test.py` provides guarded Warp
CPU parity and smoke coverage for scalar diffusion/gravity regimes and vector
lanes. Rectangular comparisons use `rtol=1e-10, atol=1e-20`; spherical
comparisons record `rtol=1.002e-3` because of the existing CPU Debye endpoint
quadrature discrepancy. P2 adds no public export, configuration/validation,
CPU physics change, charged physics, container mutation, or RNG behavior.

## Delivered Phase: E6-F3-P3 (#1403)

P3 ships the neutral direct-kernel input boundary in
`particula/gpu/kernels/wall_loss.py`: the frozen, concrete-module-only
`NeutralWallLossConfig` and write-free `wall_loss_step_gpu` preflight. The
step is lazily exported from `particula.gpu.kernels`, while the configuration
remains deliberately unexported from package namespaces.

Preflight validates neutral spherical/rectangular configuration, fixed-shape
Warp particle metadata and finite physical domains, time, direct or explicit
environment inputs, and optional RNG-sidecar metadata before any mutable
runtime work. `particula/gpu/kernels/tests/wall_loss_test.py` provides
Warp-guarded configuration, import-boundary, validation-order, and failure
atomicity coverage. P3 performs no coefficient assembly, removal execution,
RNG initialization/advancement, output allocation, or particle mutation.

## User Stories

- As a particle-resolved simulation developer, I want neutral spherical and
  rectangular wall loss on Warp so that I can keep process state device-resident.
- As a physics maintainer, I want deterministic coefficient comparisons against
  the CPU strategies and statistically bounded survival outcomes so that the GPU
  implementation is scientifically reviewable without requiring identical RNG
  sequences.
- As a workflow integrator, I want persistent caller-owned RNG and complete slot
  clearing so repeated direct steps preserve reproducibility and fixed-slot
  invariants.

## Delivered Phase: E6-F3-P4 (#1404)

P4 now executes bounded neutral particle-resolved wall loss through
`wall_loss_step_gpu`. After frozen P3 preflight, positive-time calls normalize
the environment, evaluate the spherical or rectangular coefficient for usable
slots, make one deterministic call-local seed-plus-slot draw per eligible slot,
and clear every mass lane, concentration, and charge on removal. Zero-time calls
complete preflight but make no writes. Optional `rng_states` is intentionally
neither initialized nor advanced; persistent lifecycle remains P5 work.
