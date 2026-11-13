# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| CUDA becomes accidentally required in default CI. | CPU-only CI fails and contributors without GPUs are blocked. | Medium | Keep CUDA behind `cuda_available()` checks and standardized skip helpers; test CPU-only paths with fakes. | Implementer |
| Marker names drift between `conftest.py`, `pyproject.toml`, and docs. | Unknown-marker warnings or confusing contributor guidance. | Medium | Update both config locations and docs in the same phase; add hook tests where feasible. | Implementer |
| Stochastic tests become flaky due to over-tight tolerances. | GPU kernel tests produce nondeterministic failures. | Medium | Use aggregate seed/step checks and existing `3 sigma` tolerance patterns; avoid exact per-seed equality. | Implementer/Reviewer |
| Helpers hide synchronization or fallback behavior. | Tests no longer validate true device behavior. | Low | Keep helpers limited to selection/skipping/tolerance metadata; no implicit data transfer. | Reviewer |
| Policy docs diverge from E3-F1/E3-F2 outcomes. | Future tests validate outdated RNG or sampling assumptions. | Medium | Re-check sibling feature outcomes before final docs phase ships. | Implementer |
| Too many markers make test selection confusing. | Contributors misuse or ignore markers. | Low | Keep marker vocabulary small and document examples. | Reviewer |

## Active Watch Items

- Confirm final E3-F1 RNG-state fixture names before standardizing examples.
- Confirm final E3-F2 mixed-scale tolerance conclusions before freezing policy
  language.
