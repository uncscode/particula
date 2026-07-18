# Architecture Design

## High-Level Design

```text
coagulation_step_gpu(...,
                      mechanism_config=CoagulationMechanismConfig(
                          ("turbulent_shear_st1956",)),
                      turbulent_dissipation=scalar-or-(n_boxes,),
                      fluid_density=scalar-or-(n_boxes,))
  -> canonicalize the mechanism mask and validate its structure
  -> validate particle schema/device metadata
  -> validate and normalize explicit positive-finite per-box P2 inputs
  -> reject a turbulent mixed mask after P2 validation, before mutable work
  -> validate environment, volume, outputs, and RNG
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
  `fluid_density` inputs. For a structurally valid ST1956 request, both are
  required after particle schema/device metadata validation.
- `_ensure_turbulent_input_array` in
  `particula/gpu/kernels/coagulation.py` accepts positive finite Python or
  NumPy floating scalars and broadcasts them privately on the active device, or
  returns a same-device `wp.float64` Warp array of shape `(n_boxes,)` by
  identity. It rejects missing, non-floating, non-finite, non-positive,
  wrong-shape, wrong-dtype, and wrong-device values.
- P2 performs only the particle schema/device metadata access needed for that
  validation. Invalid P2 input fails before environment/volume normalization,
  output or RNG setup, allocation, kernel launch, output writes, particle
  mutation, or RNG advancement. Valid singleton input proceeds to execution;
  valid mixed turbulent masks validate P2 first, then reject before
  normalization, allocation, RNG work, launch, or mutation.
- Non-turbulent masks do not inspect, normalize, allocate for, or reject either
  turbulence argument, preserving Brownian/charged/sedimentation behavior.

- **Data Model:** No `WarpParticleData` or `WarpEnvironmentData` schema change.
  Dissipation `[m^2/s^3]` and fluid density `[kg/m^3]` are call-specific inputs,
  normalized to active-device `wp.float64` arrays shaped `(n_boxes,)`; supplied
  valid arrays retain identity.
- **API Surface:** `coagulation_step_gpu` has the implemented keyword-only P2
  inputs. They are required only when turbulent shear is enabled and are ignored
  for non-turbulent masks. Existing positional calls and the return tuple remain
  source compatible.
- **Kernel Interface:** Use the focused ST1956 pair helper and pass normalized
  per-box arrays to the shared property/majorant/selection path. Do not add a
  separate public turbulent step or separate stochastic pass.
- **Workflow Hooks:** E5-F5 depends on E5-F1, runs independently of E5-F3/F4,
   supplies the term/majorant to E5-F6, fixtures to E5-F7, and support facts to
   E5-F9.

### Implemented P3 Execution Boundary

- The capability matrix executes only the exact ST1956 singleton. Valid mixed
  turbulent masks validate their required P2 inputs, then reject during
  preflight before normalization, allocation, RNG setup, launch, or mutation.
- The singleton uses normalized same-device fp64 per-box state, derives
  kinematic viscosity and active radii on device, and uses the ST1956 rate at
  the two largest compact active radii as an O(A) safe majorant.
- It retains the shared bounded selector, one acceptance draw per valid
  candidate, existing merge pass, caller-owned collision outputs, and persistent
  RNG behavior. Host-side preflight errors are non-mutating; runtime errors
  after launch have no rollback guarantee for caller-owned mutable state.

## Explicit Scientific Boundary

The currently supported direct-step behavior is the exact particle-resolved
ST1956 singleton with P2-normalized caller-supplied box state. This feature does
implement, validate, approximate, or claim parity with the repository's
turbulent DNS models. It makes no DNS, clustering, inertial enhancement,
general turbulence, or accuracy claim beyond the tested ST1956 formula and
bounded-sampler design.

## Security & Compliance

There is no network or authorization surface. Numerical/state robustness is the
compliance boundary: reject missing, non-finite, non-positive, wrong-shape,
wrong-dtype, wrong-device, or unsupported inputs before mutation or RNG
advancement; require finite non-negative rates and majorants; maintain
`pair_rate <= majorant` apart from a narrow defensive roundoff clamp; conserve
species mass; and perform no hidden host read, fallback, synchronization, or
transfer. Optional CUDA cannot replace required Warp CPU evidence.
