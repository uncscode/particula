# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|---|---|---|---|---|
| Concentrations are mixed directly instead of extensive amounts. | High: unequal-volume boxes violate conservation. | Medium | Stage `concentration * volume`, transfer amounts, then normalize by final volume; verify independent ledgers. | P2/P3 implementer |
| In-place edge updates make results declaration-order dependent. | High: nondeterministic scientific results. | Medium | Read only pre-node state into scratch, accumulate synchronously, and add edge-permutation tests. | P3 implementer |
| Particle transfer silently loses composition or charge. | High: corrupt aerosol state. | Medium | Move concentration with immutable mass/charge metadata, check per-species/charge ledgers, and reject unrepresentable plans. | P4 implementer |
| Fixed destination capacity is exceeded after partial mutation. | High: faulted state or silent clipping. | Medium | Complete deterministic slot planning/status validation before the single commit; never resize or truncate. | P4 implementer |
| Volume evolution double-counts dilution. | High: concentrations receive both transport and expansion scaling. | Medium | Specify one extensive-ledger transaction and canonical ordering; distinguish existing dilution sink physics from conservative transport. | Architecture reviewer |
| Communication leaves consumer-derived state stale. | High: later physics consumes inconsistent state. | Medium | P5 invalidates saturation ratio only after either barrier, preserves fresh vapor pressure, and tests the existing condensation/diagnostics consumer refresh windows. | P5 implementer |
| Validation scans or diagnostics introduce hidden host synchronization. | Medium: breaks resident-loop contract. | Medium | Keep normal status/diagnostics resident; permit host reads only at documented setup/checkpoint/error boundaries. | GPU reviewer |
| Scope expands into CFD, dynamic particles, or optimization. | High: T7 becomes unshippable. | Medium | Enforce issue #1451 prescribed-map boundary and defer CFD, graph capture/performance, and autodiff. | Feature owner |
| CUDA evidence is treated as mandatory or exact CPU replay. | Medium: CI instability and overstated guarantees. | Low | Use Warp CPU baseline, optional clean CUDA skips, and tolerance/invariant comparisons. | Test reviewer |
| Upstream E7-F4/F5/F6 contracts change while T7 is implemented. | Medium: integration churn. | Medium | Gate P5 on shipped upstream APIs and keep P1-P4 behind narrow adapter/resource seams. | Feature owner |
