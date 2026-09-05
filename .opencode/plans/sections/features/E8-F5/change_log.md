# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-09-04 | Shipped E8-F5-P3 for issue #1577: added test-only optional native-CUDA captured resident-loop parity/diagnostic validation for separate GAS and PARTICLES closed maps, active/prescribed-volume/no-work scenarios, opaque candidate qualification/rejection, and replay forbidden-host-work instrumentation in `particula/execution/tests/captured_full_loop_test.py`; no production modules, APIs, exports, user docs, examples, or architecture changed | plan-update-full |
| 2026-09-04 | Shipped E8-F5-P2 for issue #1576: added test-only E8-F2 prepared uncaptured Warp-CPU multi-timestep parity/conservation evidence, forbidden-work spies, and zero-duration coverage in `particula/execution/tests/captured_full_loop_test.py`; no production modules, APIs, scheduler behavior, or capture/replay behavior changed | plan-update-full |
| 2026-09-04 | Shipped E8-F5-P1 for issue #1575: added the test-only immutable two-box scenario, independent NumPy full-loop oracle, diagnostics/inventory assertions, no-work coverage, and validation rows in `particula/execution/tests/captured_full_loop_test.py`; no production modules or APIs changed | plan-update-full |
| 2026-08-30 | Required independent runtime qualification for every Warp-visible CUDA device; recorded the local RTX 5060 as an initial evidence device rather than a hardware restriction | user decision |
| 2026-08-30 | Initial E8-F5 three-way full-loop validation plan drafted with five issue-sized phases | plan-feature-drafter |
| 2026-08-30 | Preserved classifier diagnostics (`none`) and flagged the parent E8-F4/E8-F5 track-number mismatch | plan-feature-drafter |
