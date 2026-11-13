# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner | Status |
|------|------------|--------|------------|-------|--------|
| Charged or total majorant does not bound extreme mixed-scale rates | Medium | High | Prove bounds for approved variants; add adversarial mixed-scale and finite-value tests before enabling execution | E5-F2/F3/F6 owners | Open |
| Sequential mechanism sampling biases combined collision counts | Medium | High | Enforce additive pair rates and one acceptance/RNG pass; test two-way and four-way aggregate rates | E5-F6 owner | Open |
| Merge conserves mass but loses or duplicates charge | Medium | High | Transfer recipient-plus-donor charge, clear donor, and assert mass and charge independently in every mechanism suite | E5-F2/F7 owners | Open |
| New inputs mutate particles or RNG before validation failure | Medium | High | Centralize preflight validation and regression-test state/RNG identity on errors | E5-F1/F7 owners | Open |
| Sedimentation scope implies unfinished collision efficiency | Medium | Medium | Fix initial efficiency at 1 and state the exclusion in API and support docs | E5-F4/F9 owners | Open |
| Stochastic CPU/Warp checks are flaky or demand exact replay | Medium | Medium | Use deterministic pair parity plus seeded aggregate/sigma bounds and conservation invariants | E5-F7 owner | Open |
| CUDA-only behavior escapes routine validation | Low | High | Make Warp CPU release-blocking when installed and run optional CUDA as additive evidence with clean skips | E5-F7 owner | Open |
| Existing Brownian API, buffers, or persistent RNG regress | Medium | High | Preserve defaults and ownership; retain focused Brownian compatibility and repeated-call tests | E5-F1/F3/F7 owners | Open |
| Documentation overclaims unsupported modes or leaves E4 debt unresolved | Medium | Medium | Gate closeout on support matrix, runnable example, independent condensation walkthrough, ownership record, and roadmap link audit | E5-F8/F9 owners | Open |
