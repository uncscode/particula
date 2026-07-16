# Architecture Design

## High-Level Design

```text
coagulation_step_gpu(..., mechanisms=("turbulent_shear",),
                     turbulent_dissipation=scalar-or-(n_boxes,),
                     fluid_density=scalar-or-(n_boxes,))
  -> E5-F1 canonicalizes the mechanism mask and validates capability
  -> validate/normalize explicit positive-finite per-box inputs
  -> validate particles, environment, volume, outputs, and persistent RNG
  -> for each box on device
       mu = dynamic_viscosity_wp(temperature[box])
       nu = mu / fluid_density[box]
       derive active particle radii from species mass/density
       ST1956(i,j) = sqrt(pi*epsilon[box]/(120*nu))
                       * (2*r_i + 2*r_j)^3
       majorant = proven maximum across active unordered pairs
  -> E5-F1 bounded active-pair loop
       accept once using ST1956(i,j) / majorant
       remove accepted pair once; advance one per-box RNG stream
  -> existing apply path mutates masses/concentration once
  -> return existing particles/pairs/counts tuple
```

Because the ST1956 prefactor is constant within a box and the rate is monotone
in the radius sum, the two largest active radii provide a tight majorant. An
exhaustive active-pair maximum is also correct and may be preferred initially
for uniformity with sibling mechanisms. The implementation decision must be
covered by an independent all-pairs assertion; no unproved extrema heuristic is
acceptable.

## Data / API / Workflow Changes

- **Data Model:** No `WarpParticleData` or `WarpEnvironmentData` schema change.
  Dissipation `[m^2/s^3]` and fluid density `[kg/m^3]` are call-specific inputs,
  normalized to active-device fp64 arrays shaped `(n_boxes,)`.
- **API Surface:** Extend E5-F1's concrete mechanism configuration capability
  matrix and `coagulation_step_gpu` with keyword-only mechanism inputs. They are
  required when turbulent shear is enabled and ignored inputs should be
  rejected or governed consistently by E5-F1's excess-input policy. Existing
  Brownian calls and the return tuple remain source compatible.
- **Kernel Interface:** Add a focused ST1956 pair helper and pass normalized
  per-box arrays to the shared property/majorant/selection path. Do not add a
  separate public turbulent step or separate stochastic pass.
- **Workflow Hooks:** E5-F5 depends on E5-F1, runs independently of E5-F3/F4,
  supplies the term/majorant to E5-F6, fixtures to E5-F7, and support facts to
  E5-F9.

## Explicit Scientific Boundary

Supported behavior is the Saffman-Turner 1956 turbulent-shear collision kernel
for direct particle-resolved fp64 execution with caller-supplied box state.
This feature does not implement, validate, approximate, or claim parity with
the repository's turbulent DNS models. It makes no DNS, clustering, inertial
enhancement, general turbulence, or accuracy claim beyond the tested ST1956
formula and bounded sampler.

## Security & Compliance

There is no network or authorization surface. Numerical/state robustness is the
compliance boundary: reject missing, non-finite, non-positive, wrong-shape,
wrong-dtype, wrong-device, or unsupported inputs before mutation or RNG
advancement; require finite non-negative rates and majorants; maintain
`pair_rate <= majorant` apart from a narrow defensive roundoff clamp; conserve
species mass; and perform no hidden host read, fallback, synchronization, or
transfer. Optional CUDA cannot replace required Warp CPU evidence.
