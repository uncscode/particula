# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Broad exception handling converts a runtime defect into CPU fallback | Medium | Critical | Fallback only consumes preflight reason codes; sentinel tests prove post-invocation errors propagate | E7-F6 owner | Open |
| Fallback hides device transfer or synchronization | Medium | High | Permit only CPU-authoritative/pre-upload or explicit restored boundaries; use transfer/sync spies | E7-F6 owner | Open |
| Error taxonomy duplicates or destabilizes E7-F1 types | Medium | High | Extend E7-F1 contracts after dependency lands; one execution exception root and reason enum | E7-F1/E7-F6 owners | Open |
| Optional Warp import leaks into CPU-only execution imports | Medium | High | Dependency-neutral protocols, lazy providers, blocked-Warp subprocess tests | E7-F6 owner | Open |
| Public exports expose concrete sidecars/configuration and become permanent | Medium | Medium | Exact allowlist/denylist tests and concrete-module-only policy | API reviewer | Open |
| Experimental labeling causes import-time warnings under `-Werror` | Low | Medium | Use docs/module metadata and release notes, not unconditional warnings | E7-F6 owner | Open |
| Existing direct GPU users interpret experimental status as immediate removal | Low | Medium | No removals in T6; publish compatibility and deprecation rules | Documentation owner | Open |
| Availability probing allocates or launches work | Low | High | Provider contract is read-only/non-launching; fake-provider and launch-spy tests | E7-F6 owner | Open |
