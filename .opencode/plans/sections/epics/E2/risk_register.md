## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Environment ownership overlaps with gas vapor-pressure behavior. | Confusing APIs and schema churn. | Medium | Resolve explicitly in E2-F1 and verify in E2-F4 docs/tests. |
| Scalar GPU compatibility is broken while adding per-box state. | Existing users and tests fail. | Medium | Add compatibility tests before kernel migration changes in E2-F5. |
| Warp unavailable environments mask GPU transfer regressions. | CI misses CUDA-only issues. | Medium | Keep CPU Warp tests comprehensive and mark CUDA-specific tests by availability. |
| Numerical studies become implementation projects. | Scope expands and delays foundation delivery. | Medium | E2-F6/F7 must produce recommendations and targeted tests, not broad rewrites. |
| CPU multi-box capability is overstated. | Users assume unsupported dynamics are available. | High | E2-F8 and E2-F9 must clearly separate container support from dynamics support. |
| Gas round-trip semantics remain ambiguous. | Downstream code loses names or vapor pressure silently. | High | E2-F4 must lock behavior with explicit tests and docs. |
| Test coverage is delayed to the end. | Integration defects accumulate. | Low | Require each child track to ship tests with implementation changes. |

### Current Challenges

- The plan tooling currently rejects `add-phase` for epic records, so milestone
  groupings are documented in section files rather than stored as plan phases.
