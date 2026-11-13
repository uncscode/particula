# Architecture Design

## High-Level Design

```text
Charged-containing `coagulation_step_gpu`
  -> complete capability, buffer, RNG, and forced finite-charge preflight
  -> allocate private fp64 total_masses after preflight
  -> prepare compact active slots and sum each valid slot's species once
  -> scan compact rank pairs i < j for a finite non-negative O(A²) bound
  -> shared selector evaluates the enabled additive candidate rate once
  -> existing apply kernel merges accepted pairs and conserves/clears charge
```

The initial charged bound is an exhaustive active-pair maximum. Charged kernels
depend on radius, mass, charge sign, and charge magnitude, so assuming extrema
from the Brownian size-only proof is unsafe. The exhaustive scan is explicit,
deterministic, and reviewable. Any later optimization must retain a mathematical
proof and pairwise regression evidence. For a combined request, the same single
exhaustive active-pair scan evaluates `_total_pair_rate(actual_mask)`; its
maximum bounds the independently sanitized Brownian-plus-charged sum without
double-counting a charged-only bound.

P2 connects the exact charged-only mask to the existing selection and acceptance
pass without a mechanism-specific selector. Active-slot preparation clears and
then writes one summed species mass per valid slot into private call-local fp64
`total_masses`; the charged O(A²) scan and candidate-rate path consume that
prepared value without per-pair species reductions. The scratch is allocated
only after all caller-resource preflight succeeds and is neither returned nor
caller-owned. Zero or non-finite candidate results use the existing sanitizer.
P3 routes either normalized combined ordering through that same selector,
collision/count buffers, RNG stream, and apply kernel; it adds neither a second
selector pass nor per-mechanism buffers.

## Data / API / Workflow Changes

- **Data Model:** No particle schema changes. Continue using E5-F1's frozen
  mechanism configuration/mask and E5-F2's existing fp64
  `WarpParticleData.charge` field.
- **API Surface:** The existing keyword-only mechanism configuration permits
  Brownian-only, exact charged-hard-sphere, and canonical Brownian-plus-charged
  particle-resolved execution; its signature and three-item return tuple are
  unchanged.
- **Kernel Interface:** The existing selector receives total masses, signed
  charge, per-box thermodynamics, and physical constants for the charged rate.
  It still performs one selector and one apply launch.
- **Ownership:** `total_masses` is the sole new private scratch allocation.
  Caller collision buffers and persistent RNG retain identity and reuse rules.
- **Validation:** Every charged-containing execution forces finite-charge
  preflight even when Brownian's optional charge validation is disabled.
- **Workflow Hooks:** E5-F3 consumes E5-F1/F2; P3 extends the charged-only path
  with the shipped combined capability.

## Security & Compliance

There is no network or authorization surface. Numerical and state robustness are
the compliance boundary: reject unsupported charged models/distributions and
invalid charge/environment/buffer inputs before any launch; require finite,
non-negative rates and majorants; prevent acceptance ratios above one except for
a narrowly defensive roundoff clamp; preserve mass and charge; and perform no
hidden CPU fallback, synchronization, or device transfer. CUDA is optional and
must not weaken Warp CPU release evidence.
