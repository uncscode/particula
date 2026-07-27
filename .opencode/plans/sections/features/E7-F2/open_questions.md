# Open Questions

These questions must be resolved from the shipped E7-F1/E7-F6 contracts before
implementation; they do not authorize scope expansion.

1. **Partially resolved by P2:** concrete carrier types live at
   `particula.execution.adapters.condensation`, while `particula.execution`
   remains Warp/GPU-free on import and retains its exact ten-name selection
   export. The future E7-F1/E7-F6 registration seam remains open.
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
6. **Resolved for P2 only:** scratch may be omitted and, when supplied, is an
   opaque caller-owned reference. A selected API's stable scratch policy remains
   deferred.

Default decisions if no upstream contract says otherwise: preserve direct
kernel optionality, expose no concrete scratch type at top level, reject
non-representable physics, return truthful process-specific output metadata,
and make no hidden transfer or fallback.
