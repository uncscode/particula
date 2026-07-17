# Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-07-15 | Initial E5-F2 plan drafted with five issue-sized phases for charged pair primitives, approved model parity, charge preflight, charge-conserving merges, and development documentation. Preserved classifier diagnostics: none. | plan-feature-drafter |
| 2026-07-16 | Completed E5-F2-P1 for issue #1336: added internal scalar fp64 Warp Coulomb, reduced-property, enhancement-limit, and diffusive-Knudsen helpers with independent co-located parity probes/tests. No public API, container, Brownian-dispatch, charged-execution, or module-boundary changes. | plan-update-full |
| 2026-07-17 | Completed E5-F2-P2 for issue #1337: added the internal fp64 Warp charged hard-sphere pair helper and independent NumPy/Warp parity coverage for valid charged states, pair-order symmetry, neutral behavior, exact extreme-repulsion zero, and exhaustive safe-zero inputs. No public API, dispatch, selection, or charged-execution integration changed. | plan-update-full |
| 2026-07-17 | Completed E5-F2-P4 for issue #1339: accepted GPU coagulation merges now add donor fp64 charge to the recipient and clear the donor charge with mass and concentration. Updated direct callers and deterministic signed-charge, multi-box inventory-conservation coverage; public step API, return tuple, sidecars, and persistent RNG ownership are unchanged. | plan-update-full |
