# E3-F2 Risk Register

| Risk | Impact | Mitigation | Owner |
| --- | --- | --- | --- |
| E3-F1 RNG contract changes after E3-F2 starts | Repeated-step stochastic tests may become invalid | Sequence implementation after E3-F1 or gate RNG-sensitive checks behind the finalized contract | E3-F2 implementer |
| Fixed-bin majorants are not conservative | Statistical correctness could be biased | Compare against CPU Brownian expected rates and keep fallback to global majorant | E3-F2 implementer/reviewer |
| Diagnostics require hidden host readback | Violates GPU transfer-boundary constraints | Keep metrics test/debug-only and document explicit synchronization/readback | E3-F2 implementer |
| Stratification adds too much sequential work to one-thread-per-box kernel | Performance may regress despite higher acceptance | Keep phase scoped to small fixed bins and allow documented limitation if improvement is not justified | E3 owner |
| Stochastic tests are flaky | CI instability | Use aggregate trials, expected sigma tolerances, deterministic seeds/RNG states, and conservative thresholds | Test author |
| CUDA-specific behavior diverges from Warp CPU | Hidden portability defect | Use existing CPU plus CUDA-if-available parametrization and skip CUDA cleanly | GPU maintainer |
