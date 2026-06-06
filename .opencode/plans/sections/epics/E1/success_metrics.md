- [ ] `E1-F1` ships a public latent-heat builder, factory registration, and
  `particula.dynamics` export path without breaking existing condensation
  strategy imports.
- [ ] The final epic scope is explicit: either follow-on validation,
  documentation, and acceleration-readiness child plans are created, or the
  epic title/scope is reduced to match the delivered feature set.
- [ ] All plan JSON files for `E1` and `E1-F1` pass `adw plans validate`
  without schema drift.
- [ ] Documentation updates describe the shipped public construction flow rather
  than direct internal-only usage patterns.

**Metrics:**

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Public latent-heat builder import paths | No public builder path | Importable from `particula.dynamics.condensation` and `particula.dynamics` | Export smoke tests in `particula/dynamics/tests/condensation_exports_test.py` |
| Factory strategy coverage | No supported latent-heat factory key | `CondensationFactory.get_strategy("latent_heat", ...)` passes regression tests | `particula/dynamics/condensation/tests/condensation_factories_test.py` |
| Scoped plan validation errors | 2 schema issues were previously reported in review | 0 validation errors for scoped plans | `adw plans validate` |
| Epic scope ambiguity | Epic title implies work beyond the single child feature | 0 unresolved scope mismatches at ship time | `child_plans.md`, `milestones_timeline.md`, and PR review checklist |
