# Architecture Design

## High-Level Design

```text
P1 private dispatcher
  -> receives E5-F1 compact active indices and term mask
  -> Brownian: existing majorant path (unchanged)
  -> charged hard sphere: scan each compact rank pair i < j
       -> evaluate E5-F2 charged_hard_sphere_wp
       -> sanitize candidate; retain finite non-negative maximum
  -> return dispatcher result
```

The initial charged bound is an exhaustive active-pair maximum. Charged kernels
depend on radius, mass, charge sign, and charge magnitude, so assuming extrema
from the Brownian size-only proof is unsafe. The exhaustive scan is explicit,
deterministic, and reviewable. Any later optimization must retain a mathematical
proof and pairwise regression evidence. For a combined request, summing valid
per-term majorants safely bounds the sum of enabled pair rates.

P1 does not connect the charged term to selection or acceptance. It is read-only:
it does not allocate, mutate simulation state, access RNG state, or build the
compact active list. Zero or non-finite candidate results are handled by the
existing sanitizer. P2/P3 will connect enabled terms to one selection and
acceptance pass without creating mechanism-specific selectors.

## Data / API / Workflow Changes

- **Data Model:** No particle schema changes. Continue using E5-F1's frozen
  mechanism configuration/mask and E5-F2's existing fp64
  `WarpParticleData.charge` field.
- **API Surface:** No public API or capability-matrix change in P1;
  `coagulation_step_gpu` continues to reject charged execution.
- **Kernel Interface:** The internal majorant dispatcher gains the charged
  branch only. No selector, pair-rate execution dispatch, merge path, callback,
  or stochastic launch changes.
- **Ownership:** The helper is read-only and adds no buffer, RNG, transfer, or
  allocation ownership.
- **Workflow Hooks:** E5-F3 consumes E5-F1/F2 and supplies an internal charged
  majorant foundation to P2/P3.

## Security & Compliance

There is no network or authorization surface. Numerical and state robustness are
the compliance boundary: reject unsupported charged models/distributions and
invalid charge/environment/buffer inputs before any launch; require finite,
non-negative rates and majorants; prevent acceptance ratios above one except for
a narrowly defensive roundoff clamp; preserve mass and charge; and perform no
hidden CPU fallback, synchronization, or device transfer. CUDA is optional and
must not weaken Warp CPU release evidence.
