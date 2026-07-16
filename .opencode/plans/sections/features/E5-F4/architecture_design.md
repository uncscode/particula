# Architecture Design

## High-Level Design

```text
coagulation_step_gpu(..., mechanisms=("sedimentation",))
  -> E5-F1 canonicalizes the mechanism mask and validates capability
  -> validate particle-resolved fp64 state, environment, volume, buffers, RNG
  -> derive each active particle on device
       total_mass = sum(species_mass)
       total_volume = sum(species_mass / species_density)
       radius = cbrt(3 * total_volume / (4*pi))
       effective_density = total_mass / total_volume
       viscosity + mean_free_path -> Kn -> Cunningham slip
       settling_velocity = 2*r^2*rho*g*C_c / (9*mu)
  -> sedimentation majorant = max(pair_rate(i, j)) for active i < j
       pair_rate = pi*(r_i+r_j)^2*abs(v_i-v_j)  # efficiency = 1
  -> E5-F1 bounded active-pair loop
       accept once with pair_rate / majorant
       remove accepted active pair once; advance one RNG state
  -> existing coagulation apply path mutates masses/concentration once
  -> return existing particles/pairs/counts tuple
```

The initial majorant deliberately scans every active unordered pair. The
SP2016 term depends on both summed radius and differential settling velocity;
using only minimum/maximum radius is not a proven bound when composition changes
density. An exhaustive maximum is deterministic and safe. A later optimization
must provide a mathematical bound and regression evidence and is not required
by E5-F4.

Sedimentation properties are derived in device code from caller-owned species
masses and species densities. No per-particle density sidecar is added to
`WarpParticleData`. Zero-concentration or non-positive-mass/volume slots remain
inactive and contribute neither a property nor a pair. All caller-controlled
invalidity is rejected in host preflight before allocations or launches that
could mutate particles, output buffers, or RNG state.

## Data / API / Workflow Changes

- **Data Model:** No particle/environment schema change. Reuse fp64 masses,
  species density, concentration, per-box volume, and normalized temperature
  and pressure. Temporary radius, effective-density, settling-velocity, and
  active-index arrays remain step-local unless E5-F1 already defines reusable
  work storage.
- **API Surface:** Expand E5-F1's concrete mechanism configuration capability
  matrix to accept the canonical sedimentation-only configuration. Do not add a
  separate public sedimentation step or collision-efficiency parameter.
- **Kernel Interface:** Add focused scalar Warp helpers and a sedimentation
  branch to the shared pair-rate/majorant dispatcher. The constant efficiency
  is encoded as 1, not accepted from caller input.
- **Workflow Hooks:** E5-F4 depends on E5-F1, supplies its term and majorant to
  E5-F6, supplies fixtures to E5-F7, and supplies final support facts to E5-F9.
- **Compatibility:** Omitted mechanism configuration retains E5-F1's legacy
  Brownian behavior. Existing return values, buffer ownership, and RNG reset
  semantics remain unchanged.

## Explicit Support Limits

- Supported: direct low-level particle-resolved fp64 execution; fixed-shape
  arrays; scalar, direct `(n_boxes,)` Warp, or explicit environment inputs;
  collision efficiency exactly 1; Stokes settling with slip correction; Warp
  CPU when Warp is installed; optional CUDA when available.
- Unsupported: non-unit or dynamic collision efficiency; drag-corrected or DNS
  settling; binned/continuous-PDF distributions; high-level strategies or
  runnables; CPU fallback; hidden transfers/synchronization; dynamic slots;
  graph capture guarantees; adaptive stepping; exact CPU/Warp RNG replay.
- Brownian-plus-sedimentation and larger combinations remain unavailable until
  E5-F6 registers and validates them.

## Security & Compliance

There is no network or authorization surface. Numerical and state robustness
form the compliance boundary: validate finite positive densities, valid
environment/volume, supported mechanism/distribution, and buffer shape/dtype/
device before mutation; require finite non-negative properties, rates, and
majorants; maintain `pair_rate <= majorant` apart from a defensive roundoff
clamp; conserve species mass; and perform no hidden host read, fallback, or
transfer. Optional CUDA results cannot replace required Warp CPU evidence.
