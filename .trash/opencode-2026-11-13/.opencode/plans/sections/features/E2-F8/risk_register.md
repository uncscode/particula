# E2-F8 Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Silent CPU coagulation box-0 behavior is preserved accidentally while docs imply full support. | High: users may trust incorrect multi-box results. | Medium | Add explicit tests and prefer `ValueError` for `n_boxes != 1`. | Implementer |
| Clarification is interpreted as a breaking API change. | Medium | Medium | Keep single-box behavior unchanged; document any new multi-box errors as unsupported-path hardening. | Reviewer |
| Docs overstate future multi-box plans. | Medium | Low | Use present-tense support tables and mark all-box execution as future work. | Documentation reviewer |
| Tests become too coupled to internals. | Low | Medium | Prefer public method tests; helper tests only for small validation utilities. | Implementer |
| Sibling feature terminology changes after this plan. | Low | Medium | Cross-check E2-F1/E2-F2 final container names during implementation. | Implementer |

## Highest Priority Mitigation

The most important mitigation is preventing silent multi-box misuse. If the
implementation must choose between compatibility and correctness, require a
reviewer decision and document the chosen support boundary explicitly.
