# Open Questions

These questions must be resolved from the shipped E7-F1/E7-F6 contracts before
implementation; they do not authorize scope expansion.

1. **Partially resolved by P3:** concrete carrier and selected adapter types live at
    `particula.execution.adapters.condensation`, while `particula.execution`
    remains Warp/GPU-free on CPU import and retains its exact ten-name selection
    export. Registrations are context-local; module-global or implicit
    registration remains outside this phase.
2. Which condensation configuration names are stable public values versus
   concrete adapter details, and how are unavailable Warp CPU and CUDA devices
   represented in the E7-F6 error taxonomy?
3. **Resolved for P3:** `ExecutionResult` wraps the native backend value without
   reconstruction; the Warp value is the actual kernel tuple, including its
   caller-owned total-transfer result. Typed process-result expansion remains
   deferred.
4. **Resolved for P5:** the native legacy CPU fixture is evidence for its own
    behavior and conservation, not a Warp numerical oracle. Resident Warp uses
    a local independent NumPy float64 fixed-four-substep P2 oracle with explicit
    per-case mass and gas tolerances; no CPU/Warp numerical-equality claim is
    made.
5. Is kappa water activity the only non-ideal GPU mapping admitted in E7-F2,
   with all BAT-specific CPU configurations rejected, or is there a narrower
   representable BAT-derived configuration already covered by shipped kernels?
6. **Resolved for P2 only:** scratch may be omitted and, when supplied, is an
   opaque caller-owned reference. A selected API's stable scratch policy remains
   deferred.
7. **Resolved for P4:** selected Warp dispatch retains and forwards thermal
   sidecars by identity, but their dependency/schema validation, execution, and
   energy output remain the direct kernel's contract. CPU selected dispatch does
   not gain latent-heat support.

Default decisions if no upstream contract says otherwise: preserve direct
kernel optionality, expose no concrete scratch type at top level, reject
non-representable physics, return truthful process-specific output metadata,
and make no hidden transfer or fallback.
