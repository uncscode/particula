# Architecture Design

## High-Level Design

```text
coagulation_step_gpu(..., mechanisms=config, mechanism-specific inputs...)
  -> canonicalize config and resolve mechanism bit mask
  -> validate mask against explicit executable combination matrix
  -> validate every enabled term's required input before launch/mutation
  -> prepare shared active indices and per-particle properties once
  -> for each box, compute each enabled term's proven majorant
  -> total_majorant = sum(enabled term majorants)
  -> schedule one bounded trial count from total_majorant
  -> for each proposed active pair
       total_pair_rate = sum(enabled pair-rate terms for i, j)
       assert/guard 0 <= total_pair_rate <= total_majorant
       draw once and accept when u < total_pair_rate / total_majorant
       remove an accepted pair once from the shared active set
  -> write one collision_pairs/n_collisions output
  -> apply accepted merges once, conserving species mass and charge
  -> return the existing particles/pairs/counts tuple
```

The total bound is the sum of valid component bounds. For non-negative rates,
if `K_m(i,j) <= M_m` for every enabled mechanism `m`, then
`sum_m K_m(i,j) <= sum_m M_m`. This bound may be conservative because component
maxima can occur at different pairs, but it is safe and avoids an unproved
cross-mechanism extrema heuristic. A tighter bound is future work unless it
comes with a proof and all-pairs evidence.

The combination matrix is data-driven on the host and explicit in device mask
branches. Canonical ordering means equivalent mechanism sets resolve to the
same mask. The implementation must not interpret duplicates as weights. The
matrix retains all shipped single-term rows, preserves Brownian-plus-charged,
adds the approved two-way rows and the full four-way row, and fails closed for
any row not deliberately registered.

## Data / API / Workflow Changes

- **Data Model:** No changes to `WarpParticleData`, `WarpEnvironmentData`, or
  transfer schemas. Reuse step-local property buffers and caller-owned output/
  RNG sidecars.
- **API Surface:** Extend only E5-F1's mechanism capability matrix and existing
  keyword-only configuration/input contract on `coagulation_step_gpu`. Existing
  omitted-configuration Brownian calls and return tuple remain compatible.
- **Kernel Interface:** Pass one resolved mask into shared property, majorant,
  and pair-rate dispatch. Add enabled terms into fp64 accumulators, then invoke
  the existing acceptance and active-pair removal logic once.
- **Input Contract:** Require charge state for charged rows and E5-F5's explicit
  positive-finite per-box dissipation/fluid-density inputs whenever turbulent
  shear is enabled. Sedimentation retains efficiency 1 and existing environment
  requirements. Reject missing, invalid, or disallowed excess inputs in host
  preflight according to E5-F1's policy.
- **Workflow Hooks:** Consume E5-F3, E5-F4, and E5-F5. Supply completed additive
  behavior to E5-F7's validation matrix and E5-F9's support documentation.

## Numerical and State Invariants

- Every component and total rate/majorant is finite and non-negative.
- `total_pair_rate <= total_majorant`; only a narrowly documented roundoff clamp
  may prevent a ratio infinitesimally above one. Material violations fail/guard
  explicitly and are never silently normalized.
- Trial scheduling remains bounded by `MAX_SCHEDULED_TRIALS_PER_BOX` and output
  capacity; no unsafe integer conversion is introduced.
- One proposal consumes one acceptance draw regardless of enabled-term count.
  Persistent per-box RNG streams are never implicitly reseeded.
- Each particle can appear in at most one accepted pair per call. Merge
  application occurs once and conserves each species' mass and total charge.

## Security & Compliance

There is no network or authorization surface. Robustness is the compliance
boundary: malformed configurations, missing mechanism inputs, wrong shape/
dtype/device, non-finite values, and unsupported distributions fail before
allocation, launch, output mutation, particle mutation, or RNG advancement.
The kernel performs no hidden host read, synchronization, transfer, or CPU
fallback. Warp CPU is required evidence when Warp is installed; CUDA remains
optional and cannot replace it.
