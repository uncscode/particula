# Architecture Design

## High-Level Design

```text
coagulation_step_gpu(..., mechanisms=None)
  -> resolve None as Brownian + particle_resolved (legacy behavior)
  -> validate structural configuration
       non-empty, known, unique, particle_resolved
  -> normalize any unique user ordering to canonical mechanism order
  -> validate current executable capability matrix
  -> validate particles/environment/output buffers/persistent RNG
  -> normalize configuration to a compact mechanism bit mask
  -> prepare properties required by enabled terms
  -> compute safe total majorant = sum(term majorants)
  -> bounded active-pair loop
       total pair rate = sum(enabled pair-rate terms)
       accept once using total pair rate / total majorant
       remove accepted pair once and update RNG state once
  -> apply accepted collision buffer once
```

The host-side configuration is a frozen dataclass in the concrete coagulation
module. Its `mechanisms` tuple uses stable string identifiers for readable
Python calls; validation canonicalizes these to an internal bit mask before
launch. `distribution_type` defaults to `"particle_resolved"` and currently
accepts no other value. An omitted configuration resolves to Brownian and
therefore does not break existing calls.

Structural support and executable support are intentionally distinct. The
schema can reserve E5 mechanism identifiers, while the public step rejects a
reserved term until its owning child feature registers pair-rate, property,
majorant, and tests. Duplicate terms are errors rather than accidental weights.
Canonical ordering makes equivalent sets resolve identically.

Warp cannot dispatch Python callbacks inside a kernel, so the sampling
interface uses a compact mask and explicit term branches. Each term contributes
a finite, non-negative pair rate and a proven finite, non-negative majorant.
The dispatcher sums all enabled pair rates and term majorants. It must maintain
`0 <= total_pair_rate <= total_majorant`; a defensive guard rejects/skips
invalid device values, while host preflight handles all caller-controlled
invalidity before launch. Downstream tracks extend dispatch, not the sampling
loop.

## Data / API / Workflow Changes

- **Data Model:** Add a frozen `CoagulationMechanismConfig` and internal
  resolved configuration/mask. Keep the configuration concrete-module-only at
  `particula.gpu.kernels.coagulation`; do not modify `WarpParticleData` or
  transfer schemas.
- **API Surface:** Add a keyword-only optional mechanism configuration to
  `coagulation_step_gpu`. Do not add a higher-level export for
  `CoagulationMechanismConfig`; only `coagulation_step_gpu` remains exported
  through `particula.gpu.kernels`.
- **Combination Matrix:** P1 recognizes Brownian plus reserved E5 term names,
  but the executable matrix initially contains Brownian only. E5-F3, E5-F4,
  E5-F5, and E5-F6 expand executable rows only with their required inputs,
  majorant proof, and tests.
- **Sampling Interface:** One normalized mask, one safe total majorant, one
  active-pair candidate stream, one acceptance decision per trial, one output
  pair buffer, and one persistent RNG stream per box.
- **Workflow Hooks:** No high-level `Aerosol` or `Runnable` integration. This
  remains the direct low-level import from `particula.gpu.kernels`.

## Security & Compliance

There is no network or authorization surface. Robustness requirements are the
security boundary: reject malformed mode names, duplicates, unsupported
distributions, unavailable terms, invalid required inputs, and unsafe
majorants before mutation or RNG advancement. Do not perform hidden host reads,
CPU fallback, synchronization, or device transfer. Error messages should name
the unsupported term or distribution without exposing device memory contents.
