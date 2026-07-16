# Architecture Design

## High-Level Design

```text
coagulation_step_gpu(..., mechanisms=config)
  -> E5-F1 resolves canonical mask and validates executable capability
  -> E5-F2 validates charge and provides approved charged pair helper
  -> prepare shared per-particle properties and compact active indices
  -> enabled-term majorants
       Brownian: existing proven Brownian bound
       charged: max(charged_rate(i, j)) over every active i < j
       total: sum of enabled term majorants
  -> one bounded candidate loop per box
       pair_rate = sum(enabled term rates for candidate i, j)
       accept once when random < pair_rate / total_majorant
       remove accepted active pair once; advance one RNG state
  -> E5-F2 charge-safe apply_coagulation_kernel once
  -> return the existing particles/pairs/counts tuple
```

The initial charged bound is an exhaustive active-pair maximum. Charged kernels
depend on radius, mass, charge sign, and charge magnitude, so assuming extrema
from the Brownian size-only proof is unsafe. The exhaustive scan is explicit,
deterministic, and reviewable. Any later optimization must retain a mathematical
proof and pairwise regression evidence. For a combined request, summing valid
per-term majorants safely bounds the sum of enabled pair rates.

The device dispatcher must calculate each candidate's Brownian and/or charged
term, sum enabled terms, and perform one acceptance draw. It must not run one
selector per mechanism. Zero or non-finite term results are handled according to
the E5-F1 device guard, while all caller-controlled invalidity fails in host
preflight before output, RNG, or particle mutation.

## Data / API / Workflow Changes

- **Data Model:** No particle schema changes. Continue using E5-F1's frozen
  mechanism configuration/mask and E5-F2's existing fp64
  `WarpParticleData.charge` field.
- **API Surface:** Expand the concrete `coagulation_step_gpu` capability matrix
  to accept the canonical charged-only and Brownian-plus-charged configurations.
  Keep the return tuple and keyword-only configuration contract unchanged.
- **Kernel Interface:** Extend the shared pair-rate/majorant dispatch with a
  charged branch and pass `particles.charge` to selection. Do not add callbacks,
  a second public step, or a second stochastic launch.
- **Ownership:** Reuse caller-owned `collision_pairs`, `n_collisions`, and
  optional `rng_states` by identity. Do not add hidden host reads or transfers.
- **Workflow Hooks:** E5-F3 consumes E5-F1/F2, then supplies executable charged
  capability and a charged majorant to E5-F6 and validation fixtures to E5-F7.

## Security & Compliance

There is no network or authorization surface. Numerical and state robustness are
the compliance boundary: reject unsupported charged models/distributions and
invalid charge/environment/buffer inputs before any launch; require finite,
non-negative rates and majorants; prevent acceptance ratios above one except for
a narrowly defensive roundoff clamp; preserve mass and charge; and perform no
hidden CPU fallback, synchronization, or device transfer. CUDA is optional and
must not weaken Warp CPU release evidence.
