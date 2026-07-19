# Architecture Design

## High-Level Design

```text
coagulation_step_gpu(..., mechanism_config=config, mechanism-specific inputs...)
  -> canonicalize mechanism_config with the identifier-to-bit mapping
  -> combine enabled bits with bitwise OR, then validate mask capability
  -> capability-reject non-turbulent deferred mask 7 before metadata access
  -> obtain particle metadata, then validate enabled terms read-only when needed
  -> capability-reject turbulent deferred masks 11, 13, and 14 before runtime
     normalization/work
  -> validate the distinct executable capability boundary
  -> prepare shared active indices and per-particle properties once
   -> for each box, compute enabled fp64 pair-rate components
   -> scan active pairs for the production total majorant
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

The sum of valid component bounds is a safe conservative bound. For non-negative rates,
if `K_m(i,j) <= M_m` for every enabled mechanism `m`, then
`sum_m K_m(i,j) <= sum_m M_m`. This bound may be conservative because component
maxima can occur at different pairs, but it is safe and avoids an unproved
cross-mechanism extrema heuristic. The shipped production selector obtains its
majorant by scanning summed active-pair rates; the summed component bound remains
the proof and diagnostic reference.

P1 separates structural recognition from the executable boundary. An immutable
identifier-to-bit mapping resolves valid mechanism identifiers; canonical
ordering and bitwise-OR construction produce the mechanism mask, after which
capability validation admits four singleton masks (`1`, `2`, `4`, `8`), all six
unordered pair masks (`3`, `5`, `6`, `9`, `10`, `12`), and four-term mask `15`.
Equivalent orderings resolve identically and duplicates are never weights.
Deferred non-turbulent mask `7` raises the stable error at the capability gate
before particle metadata or enabled-term validation.
Deferred turbulent masks `11`, `13`, and `14` access metadata and validate their
enabled terms, then raise that error before runtime normalization, output/RNG
work, launch, or mutation. The approved masks enter the shared execution path.

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
  may prevent a ratio infinitesimally above one. The private acceptance guard
  permits only `rate - majorant <= 8 * eps * max(rate, majorant)`, maps that
  permitted overshoot to `1.0`, and otherwise rejects before an RNG draw,
  selector write, or active-set removal. Material violations are never silently
  normalized.
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
