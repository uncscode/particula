# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-21 | Initial E6-F5/T5 plan drafted with five issue-sized phases; fixed-shape predicates, deterministic activation, exact diagnostics, atomic validation, and E6-F6/F7/F8/F9 dependencies preserved | plan-feature-drafter |
| 2026-07-23 | Shipped E6-F5-P1 via issue #1416: added read-only CPU `get_slot_diagnostics`, its `particula.particles` export, and regression coverage for fixed `int32` diagnostics, exact invalid-state errors, and source non-mutation/fresh allocations | plan-update-full |
| 2026-07-23 | Shipped E6-F5-P2 via issue #1417: added direct-import CPU `activate_slots` with global all-or-nothing preflight, ascending-free-slot mapping, fresh `int32` activated counts, and extensive activation/atomicity regression coverage | plan-update-full |
| 2026-07-24 | Shipped E6-F5-P3 via issue #1418: added concrete, unexported direct-Warp `get_slot_diagnostics_gpu` with read-only slot classification, caller-owned `int32` free-index/count sidecars, pre-write invalid-state rejection, and direct-Warp regression coverage | plan-update-full |
| 2026-07-23 | Shipped E6-F5-P4 via issue #1419: added package-exported direct-Warp `activate_slots_gpu` with complete atomic preflight, ascending free-slot activation, caller-owned `int32` activation/diagnostic sidecars, CPU-oracle parity and rejection-atomicity coverage, optional CUDA coverage, and public-contract guidance | plan-update-full |
| 2026-07-23 | Shipped E6-F5-P5 via issue #1420: made the foundation guide authoritative for CPU/direct-Warp fixed-slot contracts, reconciled agent, roadmap, and test guidance, published focused CPU/Warp commands and downstream E6-F6/F7/F8/F9 ownership, and recorded that no user example is shipped | adw-build |
