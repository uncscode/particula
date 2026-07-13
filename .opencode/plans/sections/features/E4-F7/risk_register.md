# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Documentation is drafted against the current pre-E4 API instead of final E4-F1 through E4-F6 behavior | Medium | High | Freeze signatures and claims only after dependency tests land; derive tables from final code and focused evidence | E4-F7-P1 owner |
| Stale issue 1272 tests are deleted or weakened to permit overclaiming | Medium | High | Revise assertions deliberately; retain explicit low-level/high-level and transfer-boundary negatives | E4-F7-P1 owner |
| Example hides synchronization, transfer, CPU vapor refresh, or allocation | Medium | High | Keep every conversion/checkpoint visible; test helper calls and buffer reuse; review for host operations inside stepping | E4-F7-P2 owner |
| Gas species order or diagnostic units are ambiguous | Medium | High | Document CPU-owned names, fixed ordering, units, sign, axes, and whole-call aggregation with checked examples | E4-F7-P1 owner |
| CUDA availability is mistaken for required support | Medium | Medium | Make Warp CPU mandatory and CUDA optional/local with skip-clean commands | E4-F7-P3 owner |
| Published test commands drift from final E4 files or markers | Medium | Medium | Execute every command immediately before publication and assert referenced paths | E4-F7-P3 owner |
| Roadmap is marked shipped before gas conservation or energy evidence lands | Low | High | Gate P4 on all upstream exit criteria and focused suites | E4-F7-P4 owner |
| Duplicate contract text diverges across pages | Medium | Medium | Keep one canonical matrix/troubleshooting page and use links elsewhere | E4-F7-P4 owner |
