# Architecture Design

## High-Level Design

```text
P1 concrete-module helpers (completed in Issue #1331)
  CoagulationMechanismConfig
    -> structural resolver: default Brownian, validate, canonicalize, mask
    -> capability validator: accept Brownian; identify reserved-term owner

P2 sampler integration (completed in Issue #1332)
  brownian_coagulation_kernel(..., mechanism_mask)
    -> private additive rate/majorant dispatch
    -> one bounded sampling pass

P3 public-step integration (completed in Issue #1333)
  coagulation_step_gpu(..., *, mechanism_config=None)
    -> resolve and capability-validate before runtime access
    -> launch existing kernel with resolved mask
```

The host-side configuration is a frozen dataclass in the concrete coagulation
module. Its `mechanisms` tuple uses stable string identifiers for readable
Python calls; validation canonicalizes these to an internal bit mask before
launch. `distribution_type` defaults to `"particle_resolved"` and currently
accepts no other value. An omitted configuration resolves to Brownian and
therefore does not break existing calls.

Structural support and executable support are intentionally distinct. Issue #1331
implemented this distinction as pure host-side helpers, not public-step
preflight: the schema recognizes canonical terms and produces fixed bits `1`,
`2`, `4`, and `8`, while the capability validator accepts Brownian only.
Reserved terms identify E5-F3, E5-F4, or E5-F5 as their owner. Duplicate terms
are errors rather than accidental weights, and canonical ordering makes
equivalent sets resolve identically.

Warp cannot dispatch Python callbacks inside a kernel, so the sampling
interface uses a compact mask and explicit fixed-mask branches. Issue #1332
implemented private additive helpers in `particula.gpu.kernels.coagulation`:
they sanitize each Brownian term to finite, strictly positive `wp.float64`
values and accumulate one `K_total` and one `M_total`. Reserved bits are
deliberate no-ops. The sampler computes `M_total` once per box and `K_total`
once per valid selected candidate; it performs one acceptance draw only when
both totals are safe and `K_total <= M_total`. Invalid, zero, and underestimated
terms are skipped before acceptance, collision output, or swap-pop mutation.
Downstream tracks extend this dispatch rather than the sampling loop.

## Data / API / Workflow Changes

- **Data Model:** Add a frozen `CoagulationMechanismConfig` and internal
  resolved configuration/mask. Keep the configuration concrete-module-only at
  `particula.gpu.kernels.coagulation`; do not modify `WarpParticleData` or
  transfer schemas.
- **API Surface:** `coagulation_step_gpu` accepts keyword-only
  `mechanism_config: CoagulationMechanismConfig | None = None`. `None` selects
  Brownian particle-resolved execution. `CoagulationMechanismConfig` remains
  concrete-module-only and is not package-exported through
  `particula.gpu.kernels`.
- **Combination Matrix:** P1 recognizes Brownian plus reserved E5 term names,
  but the executable matrix initially contains Brownian only. E5-F3, E5-F4,
  E5-F5, and E5-F6 expand executable rows only with their required inputs,
  majorant proof, and tests.
- **Sampling Interface:** The private kernel receives `mechanism_mask`
  immediately before `collision_capacity`; the public step passes the
  preflight-resolved mask at this existing boundary. It has one total majorant,
  one active-pair candidate stream, one
  acceptance draw per otherwise-valid trial, one output pair buffer, and one
  persistent RNG stream per box.
- **Workflow Hooks:** No high-level `Aerosol` or `Runnable` integration. This
  remains the direct low-level import from `particula.gpu.kernels`.

## Security & Compliance

There is no network or authorization surface. Robustness requirements are the
security boundary: reject malformed mode names, duplicates, unsupported
distributions, unavailable terms, and invalid required inputs before mutation
or RNG advancement; Issue #1333 implements that pre-RNG validation. Candidate
pair selection can advance RNG before its per-candidate rate-to-majorant guard. Unsafe
candidates then receive no acceptance draw and cause no collision write/count
increment or active-set swap-pop mutation. Do not perform hidden host reads,
CPU fallback, synchronization, or device transfer. Error messages should name
the unsupported term or distribution without exposing device memory contents.
