# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-18 | Completed E5-F5-P2 for issue #1353: added keyword-only ST1956 dissipation/fluid-density P2 inputs and same-device per-box normalization, with helper, ordering, atomicity, and non-turbulent ignored-input tests. Valid inputs retain the reserved-capability failure; no turbulent execution was added. | plan-update-full |
| 2026-07-18 | Completed E5-F5-P1 for issue #1352: added internal fp64 Warp kinematic-viscosity and ST1956 pair-rate helpers with independent parity, invariant, zero-overflow-guard, and composed Sutherland-transport tests. No public API or orchestration changes. | plan-update-full |
| 2026-07-15 | Initial E5-F5 feature plan drafted with four co-tested phases, explicit per-box dissipation/fluid-density contracts, ST1956-only support, and no DNS claims; classifier diagnostics preserved as none. | plan-feature-drafter |
