# Success Criteria

- [ ] A standalone runnable walkthrough builds NumPy oracle and Warp inputs
  independently from immutable physical constants with no output aliasing.
- [ ] The required Warp CPU path executes whenever Warp is installed; missing
  Warp and unavailable CUDA outcomes are explicit and never reported as passes.
- [ ] Physics results separately compare final particle masses and gas
  concentrations against the independent fixed-four-substep oracle using
  recorded canonical fp64 per-field tolerances.
- [ ] Conservation separately checks concentration-weighted particle-plus-gas
  inventory for every box and species at `rtol=1e-12, atol=1e-30`.
- [ ] Energy separately checks signed whole-call P2-finalized transfer times
  latent heat for every box and species at `rtol=1e-12, atol=1e-18`, including
  positive condensation and negative evaporation.
- [ ] Tests prove that a passing category cannot mask a failure in either of the
  other two categories.
- [ ] The support statement remains bounded to low-level fp64 fixed-four direct
  condensation and does not claim CPU strategy/`Runnable` parity.
- [ ] Every deferred capability has a named downstream owner, entry gate, and
  explicit E5-F8 non-claim; documentation tests fail if a row is removed.
- [ ] Canonical condensation, foundations, example, and roadmap documents link
  to the walkthrough; all links and focused commands validate.
- [ ] E5-F9 receives stable artifact links and no E5-F1-F7 scope is duplicated.
- [ ] Coverage thresholds remain unchanged and all focused tests pass with
  warnings treated as errors.

## Metrics

| Metric | Baseline | Target | Source |
|--------|----------|--------|--------|
| Independently built CPU/Warp walkthroughs | 0 | 1 runnable report | New example regression test |
| Required evidence categories reported | Existing evidence distributed across tests/docs | 3/3 separately labeled and gated | Walkthrough output and category-isolation tests |
| Conservation tolerance | Existing direct-kernel `rtol=1e-12` evidence | `rtol=1e-12`, `atol=1e-30` per box/species | Walkthrough conservation result |
| Energy tolerance | Existing #1272 `rtol=1e-12`, `atol=1e-18` evidence | Same, with both signs represented | Walkthrough energy result |
| Deferred capabilities with owner and gate | Distributed prose | 100% of required rows | Ownership-record test |
| Required CUDA dependency | None | None; clean optional skip | Device-policy test |
| Broken canonical links | Unknown before implementation | 0 | Documentation validation |
