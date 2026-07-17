# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-15 | Initial E5-F4 plan drafted for T4 SP2016 sedimentation-only GPU execution with collision efficiency fixed at 1, four co-tested phases, explicit support limits, and classifier diagnostics recorded as none | plan-feature-drafter |
| 2026-07-17 | Completed E5-F4-P1 for Issue #1347: added internal fp64 Warp mixture-density, Stokes/Cunningham settling, and SP2016 pair-rate helpers with independent direct probe tests; preserved the no-dispatch/no-public-API boundary | plan-update-full |
| 2026-07-17 | Completed E5-F4-P2 for Issue #1348: added private exact sedimentation-only dispatch using an exhaustive compact active-pair majorant, shared bounded scheduler/RNG path, and cleared call-local settling-velocity scratch; retained public rejection and private mixed-mask no-op behavior | plan-update-full |
