# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-21 | Initial E6-F3/T3 feature plan drafted with seven issue-sized phases for neutral spherical/rectangular GPU wall loss, deterministic CPU coefficient parity, statistical survival validation, persistent caller-owned RNG, fixed-slot clearing, and explicit Epic G/E6 sibling boundaries | plan-feature-drafter |
| 2026-07-22 | Recorded shipped E6-F3-P1 / #1401: canonical neutral fp64 transport ownership moved to `particula.gpu.properties`, legacy dynamics definitions/re-exports removed, consumers migrated, defined slip domain behavior added, and device-only Debye/`x_coth_x` primitives validated. Recorded that wall-loss assembly/API, charged physics, and CPU changes remain deferred. | plan-update-full |
| 2026-07-23 | Recorded shipped E6-F3-P2 / #1402: added concrete internal fp64 Warp spherical and rectangular neutral Crump-Seinfeld coefficient helpers in `particula/gpu/dynamics/wall_loss_funcs.py`, plus guarded CPU/Warp parity and smoke coverage in `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`. No public export, CPU physics, validation/configuration, charged physics, container mutation, or RNG behavior changed. | plan-update-full |
