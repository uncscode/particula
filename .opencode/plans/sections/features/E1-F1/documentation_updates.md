`E1-F1-P1` did not ship external documentation changes. The implementation was
kept intentionally scoped to code and tests:

- Added docstrings in
  `particula/dynamics/condensation/condensation_builder/condensation_latent_heat_builder.py`
  for the new public builder and its fluent setters/build path
- Updated `.opencode/plans/sections/features/E1-F1/` to reflect that P1 shipped
  the builder module, builder-package export, and focused tests

Deferred documentation work:

- `docs/Features/condensation_strategy_system.md` remains a later-phase update
  after factory registration and broader namespace exports are finalized
