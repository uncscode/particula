# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Resampling biases the represented distribution | Medium | High | Define preserved moments and compare CPU/GPU behavior to an independent CPU oracle. | Physics owner | Open |
| CPU and GPU stochastic streams differ | High | Medium | Compare coefficients deterministically and outcomes with documented statistical bounds. | Test owner | Open |
| Charged wall-loss equations are ported incorrectly | Medium | High | Require coefficient parity and zero-charge/field neutral fallback before stochastic tests. | GPU owner | Open |
| Full slots silently lose nucleation inventory | Medium | High | Finalize demand before mutation, expose exact diagnostics, and test all policy combinations. | Physics owner | Open |
| Invalid GPU calls partially mutate state | Low | High | Enforce ordered preflight and explicit immutability regression tests. | GPU owner | Open |
| Scope expands into backend scheduling or performance claims | Medium | Medium | Gate reviews against explicit Epic G and later-epic boundaries. | Epic owner | Open |
| CUDA availability blocks delivery | Low | Medium | Make Warp CPU mandatory and CUDA optional with clean skips. | Test owner | Mitigated |
| Nucleation contract lacks scientific validity bounds | Medium | High | Record equations, units, citations, validity domain, and reproducible fixtures before API freeze. | Physics owner | Open |
