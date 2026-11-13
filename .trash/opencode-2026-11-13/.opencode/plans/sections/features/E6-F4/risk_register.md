# Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| Image charge is incorrectly disabled when wall potential is zero | Medium | High | Freeze semantics in P1; parity-test nonzero charge with zero potential against the CPU oracle in P2/P5 | Physics developer |
| Field magnitude, potential scaling, or charge sign diverges from CPU | Medium | High | Port helpers independently; test spherical scalar, rectangular vector, potential-only, and both charge signs before integration | GPU kernel developer |
| Charged mode perturbs neutral particles through rounding or branching | Medium | High | Use an explicit per-slot zero-charge neutral branch and compare directly to E6-F3 outputs in deterministic and stochastic fixtures | E6-F4 owner |
| Invalid charge/configuration advances RNG or partially mutates slots | Low | High | P1 staged validation and snapshots cover caller particle, charge, rectangular-field, and RNG state; retain this boundary in later physics phases | GPU maintainer |
| Extreme enhancement overflows or produces negative/nonfinite rates | Medium | High | Preserve CPU clipping, `nan_to_num`, radius/scale guards, and final nonnegative finite clipping with edge-domain tests | Physics developer |
| Stochastic tests become flaky or imply RNG-sequence parity | Medium | Medium | Predeclare binomial/sigma bounds and sample sizes; compare distributions, not NumPy/Warp draw order | Test owner |
| E6-F3 implementation changes while E6-F4 is built | Medium | Medium | Treat E6-F3 as a hard dependency; rebase on its final API and run its full regression matrix in every E6-F4 phase | E6-F4 owner |
| Scope expands into activation, scheduling, or broader electrostatics | Medium | Medium | Enforce E6-F5/F6 and Epic G boundaries in API review, docs, and acceptance criteria | Parent E6 owner |
| CUDA absence blocks delivery | Low | Medium | Require Warp CPU; make CUDA parameterized evidence optional with clean skip behavior | Test owner |
