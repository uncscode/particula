# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-08 | Initial feature plan drafted for E3-F1 with four phases covering RNG API compatibility, persisted-state tests, implementation, and documentation. | plan-feature-drafter |
| 2026-07-08 | Updated E3-F1 sections after issue #1236 shipped P1 compatibility work: `coagulation_step_gpu` gained keyword-only `initialize_rng`, compatibility tests were added, and broader docs/benchmark updates were explicitly deferred. | plan-update-full |
| 2026-07-08 | Updated E3-F1 sections after issue #1237 shipped P2 as a test-only regression phase: the repeated valid-call test was renamed to make the persisted caller-owned buffer contract explicit, and a valid-then-invalid `time_step` regression was added to prove an already-advanced buffer is preserved on early failure. | plan-update-full |
| 2026-07-09 | Updated E3-F1 sections after issue #1239 shipped P4 as a docs-only phase: the `coagulation_step_gpu` docstring, roadmap guidance, and container-boundary guide now consistently describe seed-once repeated-call usage, caller-owned persistent `rng_states`, and pre-capture initialization/reset expectations with no runtime behavior changes. | plan-update-full |
