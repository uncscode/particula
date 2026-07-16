# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Configuration shape becomes incompatible with later per-box mechanism inputs | Medium | High | Keep term selection as host metadata and mechanism-specific arrays as explicit keyword/sidecar inputs; review P1 with E5-F4/F5 requirements | E5-F1 |
| Reserved identifiers are mistaken for executable support | Medium | High | Separate structural and executable capability tables; reject unavailable terms and label downstream owners in errors/docs | E5-F1 |
| Additive refactor changes Brownian stochastic behavior | Medium | High | Preserve candidate order and RNG calls; require omitted-vs-explicit seeded equivalence before P2/P3 completion | E5-F1-P2 |
| Duplicate mechanisms silently double-weight rates | Medium | High | Reject duplicates during pure host validation rather than deduplicating | E5-F1-P1 |
| An unsafe term majorant yields acceptance ratios above one | Medium | Critical | Require every capability registration to provide a proven bound; sum term bounds and add deterministic boundary tests/defensive guards | E5-F3/E5-F6 |
| Validation occurs after output or RNG mutation | Low | High | Resolve all caller-controlled configuration and required-input errors before allocation or launch; snapshot all caller-owned fields in tests | E5-F1-P3 |
| Distribution type is inferred incorrectly from fixed-shape data | Medium | Medium | Require an explicit configuration declaration and accept only `particle_resolved`; do not infer from concentration values | E5-F1-P1 |
| Warp branching or generic dispatch causes performance regression | Low | Medium | Use a compact resolved bit mask and explicit Warp branches; defer performance redesign and benchmark only for regression evidence | E5-F6 |
| Nested codebase research remains unavailable | High | Low | Use the authoritative E5 research references and record the tooling limitation; require implementation review against current files | Plan drafter |
