# Success Criteria

## Pass / Fail Criteria

- [ ] A new default integration test exercises `CondensationLatentHeat`
  end-to-end through the CPU `MassCondensation` runnable path.
- [ ] The baseline fixture starts with a minimal deterministic single-species
  case; any multi-species extension remains optional and ships only if it stays
  fast and stable in the default integration suite.
- [ ] The test is deterministic, fast, and does not require GPU, CUDA, Warp,
  `slow`, or `performance` markers.
- [ ] Assertions prove particle water mass increases, gas water concentration
  decreases, and total water inventory remains conserved within the documented
  stable tolerance.
- [ ] `last_latent_heat_energy` is finite, positive, and matches transferred
  mass times the constant latent-heat strategy within a tight stable tolerance.
- [ ] Documentation identifies the baseline as CPU-only reference evidence for
  future Epic D GPU latent-heat parity rather than claiming shipped GPU parity.
- [ ] No production GPU runtime behavior changes are introduced while adding the
  reference baseline.

## Validation Evidence

Implementation should record focused and default integration validation results,
including at minimum:

```bash
pytest particula/integration_tests/condensation_latent_heat_conservation_test.py -q
pytest particula/integration_tests -q
```

## Evidence Metrics

| Metric | Completion Signal | Evidence Source |
| --- | --- | --- |
| Deterministic baseline fixture | Single-species CPU fixture passes consistently in default integration runs | `condensation_latent_heat_conservation_test.py` |
| Conservation | Particle + gas water inventory closes within the selected tolerance | Integration assertions from P2 |
| Latent-heat bookkeeping | `last_latent_heat_energy` matches transferred-mass calculation | Integration assertions from P2 |
| Runtime suitability | New test remains in the default integration suite without slow/performance markers | `pytest particula/integration_tests -q` |
| Scope control | Docs describe CPU-only reference status for future Epic D work | Roadmap/feature docs diff |

## Definition of Done

Reviewers can run the default CPU integration suite and point to one stable
latent-heat conservation test that serves as future GPU-parity reference
evidence.
