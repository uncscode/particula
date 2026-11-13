# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-15 | Initial E5-F4 plan drafted for T4 SP2016 sedimentation-only GPU execution with collision efficiency fixed at 1, four co-tested phases, explicit support limits, and classifier diagnostics recorded as none | plan-feature-drafter |
| 2026-07-17 | Completed E5-F4-P1 for Issue #1347: added internal fp64 Warp mixture-density, Stokes/Cunningham settling, and SP2016 pair-rate helpers with independent direct probe tests; preserved the no-dispatch/no-public-API boundary | plan-update-full |
| 2026-07-17 | Completed E5-F4-P2 for Issue #1348: added private exact sedimentation-only dispatch using an exhaustive compact active-pair majorant, shared bounded scheduler/RNG path, and cleared call-local settling-velocity scratch; retained public rejection and private mixed-mask no-op behavior | plan-update-full |
| 2026-07-17 | Completed E5-F4-P3/P4 for Issue #1349: shipped public particle-resolved `("sedimentation_sp2016",)` direct-kernel execution, sedimentation domain preflight, caller-owned output/persistent-RNG state safety, multi-box conservation regressions, and canonical user-facing contract documentation | plan-update-full |
| 2026-07-17 | Completed Issue #1350 documentation follow-up: added user-facing SP2016 equation/citation, m³/s units, fixed-unit-efficiency scope, sedimentation-focused evidence wording, and confirmed direct-input, fp64-scratch, ownership, preflight, and device boundaries without replacing #1349 or delivering E5-F6/E5-F7/E5-F9 work | adw-build |
