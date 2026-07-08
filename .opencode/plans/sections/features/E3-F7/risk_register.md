# Risk Register

| Risk | Impact | Likelihood | Mitigation | Owner |
| --- | --- | --- | --- | --- |
| Fixture becomes numerically flaky due to very small transfer amounts. | Default integration tests may fail intermittently. | Medium | Choose deterministic initial conditions with measurable condensation; prefer fixture tuning over loose tolerances. | Implementer |
| Energy units are misinterpreted as density rather than total energy. | Baseline could encode the wrong Epic D target. | Medium | Compare directly against `last_latent_heat_energy` semantics and `mass_transfer * latent_heat`; document the unit assumption. | Implementer/Reviewer |
| Documentation implies GPU latent-heat parity. | Users may rely on unsupported behavior. | Low | Use explicit CPU-only wording and state GPU parity remains future Epic D work. | Documentation reviewer |
| Test duplicates too much setup from existing integration tests. | Maintenance burden increases. | Medium | Keep local helpers small; only extract shared helpers if reuse is clearly beneficial. | Implementer |
| Test runtime grows beyond default-suite expectations. | CI and local feedback slow down. | Low | Use small deterministic particle count and short time loop; avoid performance benchmarking. | Implementer |
