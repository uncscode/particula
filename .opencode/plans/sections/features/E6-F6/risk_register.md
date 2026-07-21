# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| Resampling biases size or composition distributions | High | Medium | Freeze preserved moments and deterministic tie breaks in P1; test mixed-scale fixtures against an independent oracle. | Physics lead |
| Partial resampling writes before a later failure | High | Medium | Separate read-only all-box planning from one commit boundary; snapshot every caller-owned input/output on failure. | CPU/GPU implementer |
| Volume scaling changes physical inventory through wrong weight transform | High | Medium | Define `V*weight` invariants and allowed factors first; require CPU/Warp per-box oracle tests. | Physics lead |
| Precedence drifts between CPU and Warp | High | Low | Use one truth table and parity fixtures; assert policy code and scale diagnostics exactly. | GPU implementer |
| Demand is rounded or silently truncated | Critical | Medium | Track requested and admitted demand explicitly; success requires equality, otherwise fail closed. | Feature owner |
| Mixed particle scales hide conservation loss | High | Medium | Check every box/species and small-particle moments separately with explicit tolerances. | Test owner |
| Caller-owned scratch is undersized or on the wrong device | Medium | Medium | Validate shape/dtype/device before output clearing or kernel launch. | GPU implementer |
| Scope expands into nucleation physics or Epic G scheduling | Medium | Medium | Preserve E6-F7/F8 and Epic G boundaries in APIs, tests, and docs. | E6 owner |
| CUDA absence blocks completion | Low | Medium | Require Warp CPU evidence and make CUDA parametrization skip cleanly. | Test owner |
