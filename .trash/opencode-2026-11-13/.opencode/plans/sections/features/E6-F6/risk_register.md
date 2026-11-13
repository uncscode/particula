# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
|------|--------|------------|------------|-------|
| Resampling biases size or composition distributions | High | Medium | In P2, define preserved moments and deterministic tie breaks; test mixed-scale fixtures against an independent oracle. | Physics lead |
| Partial resampling writes before a later failure | High | Medium | Separate read-only all-box planning from one commit boundary; snapshot every caller-owned input/output on failure. | CPU/GPU implementer |
| Volume scaling changes intensive state through a wrong raw-weight/source transform | High | Medium | Require `V_new=sV`, `w_new=sw`, `E_new=sE`, and CPU/Warp per-box intensive-oracle tests. | Physics lead |
| Precedence drifts between CPU and Warp | High | Low | Use one truth table and parity fixtures; assert policy code and scale diagnostics exactly. | GPU implementer |
| Demand is rounded or silently truncated | Critical | Medium | Track potential, gas-admitted, represented, reduction, and residual demand explicitly; final-domain residual must be zero. | Feature owner |
| Mixed particle scales hide conservation loss | High | Medium | In P2/P6, check every box/species and small-particle moments separately with explicit tolerances. | Test owner |
| Caller-owned scratch is undersized or on the wrong device | Medium | Medium | Validate shape/dtype/device before output clearing or kernel launch. | GPU implementer |
| Scope expands into nucleation physics or Epic G scheduling | Medium | Medium | Preserve E6-F7/F8 and Epic G boundaries in APIs, tests, and docs. | E6 owner |
| CUDA absence blocks completion | Low | Medium | Require Warp CPU evidence and make CUDA parametrization skip cleanly. | Test owner |
