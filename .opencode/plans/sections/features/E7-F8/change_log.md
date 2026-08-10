# Change Log

| Date | Change | Author |
|---|---|---|
| 2026-07-27 | Initial E7-F8 plan drafted from issue #1451 Track T8, parent E7, E7-F3/E7-F4/E7-F5 handoffs, existing direct coagulation/wall-loss RNG contracts, and checkpoint/restart validation requirements | plan-feature-drafter |
| 2026-08-09 | Issue #1520 completed E7-F8-P1: added direct-only `particula.execution.rng` host registration, deterministic FNV derivation, lazy validated caller-buffer initialization, focused tests, and export-denial coverage; integration phases remain deferred | plan-update-full |
| 2026-08-09 | Issue #1521 completed E7-F8-P2: added concrete resident stream metadata, one first-acquisition P1-derived coagulation sidecar retained by identity, forced-false resident Brownian dispatch, and fail-closed no-persistence checkpoint/finalize handling; documented bounded ownership and unsupported scope | plan-update-full |
| 2026-08-09 | Issue #1522 completed E7-F8-P3 (commit `ca21d45d8`): published an independent canonical-manifest wall-loss sidecar with exact resource identity/nonaliasing, threaded scheduler-resolved selected-box gating through the adapter without changing the direct kernel contract, preserved disabled/prelaunch-skipped/zero-time/no-work lanes, and retained checkpoint guard and scheduler fault lifecycle regressions | plan-update-full |
