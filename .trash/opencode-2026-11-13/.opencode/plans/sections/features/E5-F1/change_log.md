# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-15 | Initial E5-F1 plan drafted from T1, parent E5 context, and authoritative issue #1320 code references; classifier diagnostics preserved as none | plan-feature-drafter |
| 2026-07-15 | Added four issue-sized phases with co-located tests and a final development-documentation phase | plan-feature-drafter |
| 2026-07-16 | Issue #1331 completed E5-F1-P1: added concrete-module-only host configuration, canonical fixed-mask resolution, Brownian-only capability validation, and co-located host tests; public-step signature and GPU runtime remain unchanged | plan-update-full |
| 2026-07-16 | Issue #1332 completed E5-F1-P2: added private Brownian additive rate/majorant mask dispatch, one-total/one-acceptance-draw sampling guards, and co-located focused deterministic and stochastic Warp tests; reserved flags remain no-ops and the public step signature remains unchanged | plan-update-full |
| 2026-07-16 | Issue #1333 completed E5-F1-P3: exposed keyword-only `mechanism_config` on `coagulation_step_gpu`, preflighted configuration before runtime access/mutation, dispatched the resolved mask at the existing kernel boundary, and added Warp CPU failure-atomicity, ordering, and explicit-versus-omitted Brownian equivalence coverage | plan-update-full |
| 2026-07-16 | Issue #1334 completed E5-F1-P4 as documentation-only work: published the Brownian-only developer contract, host-configuration/Warp-sidecar ownership boundary, executable/reserved matrix, and one-pass extension rule without changing container schemas or runtime behavior. No validation run is claimed. Brownian remains the sole executable mechanism; the complete end-user example and final support matrix remain deferred to E5-F9. | plan-update-full |
