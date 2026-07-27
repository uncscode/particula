# Open Questions

These questions must be resolved from the shipped E7-F1/E7-F6 contracts before
implementation; they do not authorize scope expansion.

1. What exact module and registration seam do E7-F1 and E7-F6 expose for a
   Warp-only adapter while keeping `particula.execution` importable without
   Warp?
2. Which condensation configuration names are stable public values versus
   concrete adapter details, and how are unavailable Warp CPU and CUDA devices
   represented in the E7-F6 error taxonomy?
3. Does E7-F1's `ExecutionResult` directly carry the kernel's total transfer, or
   should it reference a typed process result while mutation metadata remains
   generic?
4. Which existing CPU condensation fixture is the canonical scientific oracle
   for the first selected isothermal/latent-heat workflow, and what justified
   parity tolerances apply given different substep algorithms?
5. Is kappa water activity the only non-ideal GPU mapping admitted in E7-F2,
   with all BAT-specific CPU configurations rejected, or is there a narrower
   representable BAT-derived configuration already covered by shipped kernels?
6. Should optional scratch omission be supported by the first selected API, or
   should stable caller-owned scratch be required in preparation for E7-F4?

Default decisions if no upstream contract says otherwise: preserve direct
kernel optionality, expose no concrete scratch type at top level, reject
non-representable physics, return truthful process-specific output metadata,
and make no hidden transfer or fallback.
