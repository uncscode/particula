# Architecture Design

## High-Level Design

```text
coagulation_step_gpu(..., mechanisms=("turbulent_shear",),
                     turbulent_dissipation=scalar-or-(n_boxes,),
                     fluid_density=scalar-or-(n_boxes,))
  -> canonicalize the mechanism mask and validate its structure
  -> validate/normalize explicit positive-finite per-box P2 inputs
  -> unchanged reserved-capability gate (no execution)
  -> [P3 deferred] validate particles, environment, volume, outputs, and RNG
  -> for each box on device
       mu = dynamic_viscosity_wp(temperature[box])
       nu = mu / fluid_density[box]
       derive active particle radii from species mass/density
       ST1956(i,j) = sqrt(pi*epsilon[box]/(120*nu))
                       * (2*r_i + 2*r_j)^3
       majorant = ST1956 rate for the two largest active radii
  -> E5-F1 bounded active-pair loop
       accept once using ST1956(i,j) / majorant
       remove accepted pair once; advance one per-box RNG stream
  -> existing apply path mutates masses/concentration once
  -> return existing particles/pairs/counts tuple
```

Because the ST1956 prefactor is constant within a box and the rate is monotone
in the radius sum, the two largest active radii provide the required tight
majorant. Tests must exhaustively compare every active unordered pair against
this proved bound; no alternate unproved extrema heuristic is acceptable.

## Data / API / Workflow Changes

### Implemented P1 Physics Boundary

- `kinematic_viscosity_wp(dynamic_viscosity, fluid_density)` and
  `turbulent_shear_st1956_pair_rate_wp(radius_i, radius_j,
  turbulent_dissipation, kinematic_viscosity)` are internal typed fp64
  `@wp.func` helpers in `particula/gpu/dynamics/coagulation_funcs.py`.
- The ST1956 helper returns exact finite zero when dissipation is zero before
  forming the cubic diameter sum, preventing a `0 * inf` result for finite
  extreme radii. Public-input validation remains deferred to P2.
- No public export, API, mechanism capability, sampler, container, or CPU
  fallback changed in P1. P2/P3 remain responsible for connecting these pure
  helpers to the direct execution path.

### Implemented P2 Direct-Step Boundary

- `coagulation_step_gpu` has keyword-only `turbulent_dissipation` and
  `fluid_density` inputs. For a structurally valid ST1956 mask, both are
  required before the existing reserved-capability gate.
- `_ensure_turbulent_input_array` in
  `particula/gpu/kernels/coagulation.py` accepts positive finite Python or
  NumPy floating scalars and broadcasts them privately on the active device, or
  returns a supported floating same-device Warp array of shape `(n_boxes,)` by
  identity. It rejects missing, non-floating, non-finite, non-positive,
  wrong-shape, wrong-dtype, and wrong-device values.
- P2 performs only the particle schema/device metadata access needed for that
  validation. Invalid P2 input fails before environment/volume normalization,
  output or RNG setup, allocation, kernel launch, output writes, particle
  mutation, or RNG advancement. Valid input then reaches the unchanged
  `reserved for E5-F5` capability error; P3 dispatch and sampling are deferred.
- Non-turbulent masks do not inspect, normalize, allocate for, or reject either
  turbulence argument, preserving Brownian/charged/sedimentation behavior.

- **Data Model:** No `WarpParticleData` or `WarpEnvironmentData` schema change.
  Dissipation `[m^2/s^3]` and fluid density `[kg/m^3]` are call-specific inputs,
  normalized to active-device arrays shaped `(n_boxes,)`; supported supplied
  `wp.float32` and `wp.float64` arrays retain identity.
- **API Surface:** `coagulation_step_gpu` has the implemented keyword-only P2
  inputs. They are required only when turbulent shear is enabled and are ignored
  for non-turbulent masks. Existing positional calls and the return tuple remain
  source compatible.
- **Kernel Interface:** Add a focused ST1956 pair helper and pass normalized
  per-box arrays to the shared property/majorant/selection path. Do not add a
  separate public turbulent step or separate stochastic pass.
- **Workflow Hooks:** E5-F5 depends on E5-F1, runs independently of E5-F3/F4,
  supplies the term/majorant to E5-F6, fixtures to E5-F7, and support facts to
  E5-F9.

## Explicit Scientific Boundary

The currently supported direct-step behavior is P2 validation of
caller-supplied ST1956 box state followed by the reserved-capability failure.
It does not execute the turbulent-shear collision kernel. This feature does not
implement, validate, approximate, or claim parity with the repository's
turbulent DNS models. It makes no DNS, clustering, inertial enhancement,
general turbulence, or accuracy claim beyond the tested ST1956 formula and
deferred bounded-sampler design.

## Security & Compliance

There is no network or authorization surface. Numerical/state robustness is the
compliance boundary: reject missing, non-finite, non-positive, wrong-shape,
wrong-dtype, wrong-device, or unsupported inputs before mutation or RNG
advancement; require finite non-negative rates and majorants; maintain
`pair_rate <= majorant` apart from a narrow defensive roundoff clamp; conserve
species mass; and perform no hidden host read, fallback, synchronization, or
transfer. Optional CUDA cannot replace required Warp CPU evidence.
