# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Saturation-ratio species dimension conflicts with downstream expectations | Medium | Medium | Use canonical `saturation_ratio` shaped `(n_boxes, n_species)` and keep tests aligned with the E2-F1 schema foundation | P1 implementer |
| Scope expands into process or GPU migration | High | Medium | Keep this feature limited to CPU container, exports, tests, and docs; defer kernels to sibling tracks | Feature owner |
| Validation is too strict for future supersaturation use cases | Medium | Medium | Do not cap `saturation_ratio` at 1.0; require only finite nonnegative values | P1 implementer |
| Existing scalar process docs become confusing | Medium | Low | Document the transition boundary explicitly in P3 | Documentation owner |
| Copy semantics accidentally share arrays | Medium | Low | Add copy independence tests mutating original and copied arrays | P2 implementer |

## Highest Priority Mitigation

Use the E2-F1 canonical `saturation_ratio` terminology before implementation.
That decision controls validation bounds, doc wording, and downstream GPU schema
compatibility.
