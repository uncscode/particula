# Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-26 | Initial E7-F7/T7 plan drafted with seven issue-sized phases for fixed-shape map validation, volume evolution, conservative gas communication, fixed-capacity particle transport, resident scheduler integration, parity/conservation evidence, and final documentation. Preserved issue #1451 scope, E7-F4/E7-F5/E7-F6 dependencies, classifier diagnostics `none`, and explicit CFD/graph-capture/autodiff exclusions. | plan-feature-drafter |
| 2026-08-08 | Shipped E7-F7-P1 for issue #1507: added concrete-only, unexported `particula.execution.communication` fixed-shape declarations and a write-free Warp validator for metadata, schemas, aliases, domains, topology, and duplicate directed edges. Recorded that P1 retains arrays by identity and defers inventory/time-step-dependent outbound overdraw to P3. | plan-update-full |
