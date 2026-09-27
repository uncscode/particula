# Phase Details

All six phases are Not Started. Each is a bounded reviewable PR with tests in
the owning change. P1 requires M2 completed and validated; every later phase
requires its predecessor's validated gate. No separate unit-testing phase.

- [ ] **E9-M3-P1: Complete native condensation admission and configuration with scientific tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Accept M1/M2 contracts and seed scientific/compatibility ledgers.
    Complete native configuration resolution, physical state admission and
    species alignment in `condensation_strategies.py`. Carry explicit activity,
    surface and vapor-pressure configuration through existing strategy
    builders/factories where needed; do not put strategies on containers.
  - Tests: Migrate direct native setup in
    `condensation/tests/condensation_strategies_test.py`; retain missing-strategy,
    vapor-pressure, skip-index and multi-box rejection assertions. Add mixed
    partitioning, wrong mapping, direct mutation and rejection snapshots.
  - Gate: Native rate/configuration and affected builder/factory tests pass
    without facade construction or changes to scientific formulas.

- [ ] **E9-M3-P2: Complete native condensation stepping with conservation and stability fixtures**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Finish isothermal, staggered and latent-heat native stepping using
    existing mass-transfer logic; preserve sequential current-gas coupling,
    theta selection, uptake/evaporation limiting and returned state identities.
  - Tests: Migrate `condensation_strategies_test.py`,
    `staggered_mass_conservation_test.py`, `staggered_stability_test.py` and
    affected `mass_transfer_test.py`/`mass_transfer_utils_test.py` fixtures.
    Cover zero time, depleted gas, inactive particles, multiple species,
    skip/nonpartitioning lanes, latent heat and `update_gases=False` separately.
    Use non-unit volumes, unequal weights and independent weighted inventories;
    retain existing justified numerical/stability tolerances.
  - Gate: Condensation scientific tests pass; all enduring assertions mapped,
    no hidden normalization or new solver/accuracy claim.

- [ ] **E9-M3-P3: Complete native coagulation kernels and rates with scientific fixtures**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Complete ABC radius/mass/concentration/density/charge helpers and
    native kernel/rate consumers for Brownian, charged, sedimentation,
    turbulent-shear, turbulent-DNS and combined strategies. Preserve process
    controls and existing discrete/continuous-PDF integration rules.
  - Tests: Migrate `coagulation_strategy/tests/`, including
    `coagulation_strategy_abc_test.py`, `brownian_coagulation_strategy_test.py`,
    `charged_coagulation_strategy_test.py`,
    `sedimentation_coagulation_strategy_test.py`, both turbulent strategy tests
    and `combine_coagulation_strategy_test.py`. Cover non-unit volumes,
    charge/density, independent rates and single-box rejection.
  - Gate: Every supported mechanism/rate row passes without facade setup;
    combined-rate/scientific helper expectations remain independent.

- [ ] **E9-M3-P4: Complete native coagulation stepping with weighted conservation tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Finish native discrete/PDF updates and resolved binning, selection
    and merge paths in the ABC and `particle_resolved_step/` consumers. Reuse
    existing algorithms; audit temporary binned data separately from
    authoritative particles rather than assuming unit/equal weights.
  - Tests: Extend ABC and adjacent resolved-step tests with multi-species
    weighted mass conservation, non-unit volumes, supported charge bookkeeping,
    zero/one active particle, empty/zero-work cases and protected state. Use
    controlled draws for deterministic branches and aggregate bounds for
    stochastic laws; retain current random-state semantics.
  - Gate: Native stepping/scientific groups pass without relaxed conservation
    or an invented cross-backend seeded-equality requirement.

- [ ] **E9-M3-P5: Migrate neutral and charged wall-loss strategies with both CPU suites**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Replace native-path facade getters/mutators in
    `wall_loss/wall_loss_strategies.py` with approved quantities/writes.
    Preserve spherical/rectangular neutral/charged paths, binned/PDF loss,
    resolved removal, image-charge enhancement, electric-field behavior,
    coefficient helpers, stochastic controls and existing CPU semantics.
  - Tests: Migrate BOTH `dynamics/tests/wall_loss_strategies_test.py` and
    `dynamics/wall_loss/tests/wall_loss_strategies_test.py`. Retain geometry,
    distribution, empty-input, zero-charge/field neutral limits and helper
    parity. Add n_boxes!=1 rejection, non-unit-volume sink accounting,
    nonnegative concentration, selected mass effects and survivor invariants.
  - Gate: Explicitly run both suites separately. The adjacent suite is
    excluded recursively; the full runner alone cannot establish this gate.
    Do not substitute GPU laws or slot behavior for existing CPU science.

- [ ] **E9-M3-P6: Update development documentation**
  - Issue: TBD | Size: XS | Status: Not Started
  - Work: Document native scientific calls, units, mapping, process-owned
    configuration and mutation boundaries in narrow developer guidance, e.g.
    `docs/Features/condensation_strategy_system.md` where applicable. Finalize
    ledgers for M4; broad tutorials/notebooks stay M5-owned.
  - Tests: Add/retain data-only integration fixtures under
    `particula/integration_tests/` using M2 native construction and explicit
    strategy calls, not M4 runnable rewrites. Rerun focused science, BOTH
    wall-loss suites and retained transfer/export checks, then untargeted
    coverage and repository lint/type/docs checks.
  - Gate: Final revision/date, literal results/skips, compatibility inventory
    and reviewer approval establish M3 completed-and-validated. Required
    unavailable checks or unresolved scientific decisions block M4.
