# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Humidity/saturation field semantics conflict with E2-F1 schema decisions | Medium | Medium | Confirm E2-F1 naming and bounds before coding P1; keep tests aligned with schema foundation | P1 implementer |
| Scope expands into process or GPU migration | High | Medium | Keep this feature limited to CPU container, exports, tests, and docs; defer kernels to sibling tracks | Feature owner |
| Validation is too strict for future supersaturation use cases | Medium | Medium | Distinguish relative humidity from saturation ratio; only cap at 1.0 when semantics require it | P1 implementer |
| Existing scalar process docs become confusing | Medium | Low | Document the transition boundary explicitly in P3 | Documentation owner |
| Copy semantics accidentally share arrays | Medium | Low | Add copy independence tests mutating original and copied arrays | P2 implementer |

## Highest Priority Mitigation

Resolve the humidity/saturation terminology from E2-F1 before implementation.
That decision controls validation bounds, doc wording, and downstream GPU schema
compatibility.
