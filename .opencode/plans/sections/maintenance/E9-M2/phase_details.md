# Phase Details

The approved API and storage contract is E9 appendix D1–D4. Each initializer
sets the shared distribution_type explicitly: density inputs multiply by V,
raw counts do not, and continuous_pdf is radius-based dN/dr. M1's handoff must
make this unambiguous before any native construction is published.

All six phases are Not Started. Each is a bounded PR with tests in the owning
phase; there is no standalone unit-testing phase. P1 requires M1 completed and
validated, and each subsequent phase requires its predecessor completed and
validated. Required failures or unavailable evidence block the next gate.

- [ ] **E9-M2-P1: Introduce flat three-container Aerosol with migrated construction tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Entry: Accept M1 final-revision contracts, helpers and approval; resolve
    the transitional compatibility decision before changing aggregate shape.
  - Work: Implement native constructor/whole-container access in `aerosol.py`
    using M1 read-only validation and approved names. Migrate aggregate
    construction/string-representation behavioral tests. Inventory affected
    legacy consumers and any minimal approved temporary seam, owned for M6
    removal; preserve later-track execution without migrating it here.
  - Tests: Exact held identities, direct current-data visibility, mixed gas,
    wrong types/box/species layouts, unchanged candidate inputs on rejection,
    and no facade construction on the native path.
  - Gate: Native aggregate tests and affected retained consumer smoke tests
    pass; no duplicated state/physics authority or unowned temporary seam.

- [ ] **E9-M2-P2: Implement individual and coordinated replacement with atomic rejection tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Implement all three individual replacements and one coordinated
    operation with complete preflight before reference publication. Reuse M1
    validators, not coercing constructors or chained mutating public setters.
  - Tests: Each accepted candidate retained by identity; unaffected references
    stay identical; same-object assignments; individual mismatch rejects;
    valid coordinated species-layout change succeeds where individual changes
    cannot. Inject failures at each preflight stage and compare all old and
    candidate references, arrays and metadata. Cover no GPU/resident rebinding.
  - Gate: Full replacement matrix passes, including late-candidate rejection,
    with no partial state change or accidental validation side effects.

- [ ] **E9-M2-P3: Migrate aggregate construction builders with alignment and identity tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Migrate `AerosolBuilder` to the three native containers. Reuse
    `ParticleDataBuilder`, `GasDataBuilder` and explicit `EnvironmentData`
    construction; add only demonstrated construction gaps. Remove native-path
    reliance on facade strategy-name checks, not the M6-owned legacy classes.
    Freeze the capability ledger for direct mass/radius inputs and presets.
  - Tests: Migrate `aerosol_builder_test.py` behavior: fluent assembly, missing
    components, shared validation, identity, repeated builds per approved
    ownership rules, mixed/all-false gas masks and rejection nonmutation.
  - Gate: Builder and direct-constructor admission agree; no species omitted,
    no physics configuration stored, every retained capability has P4/P5 owner.

- [ ] **E9-M2-P4: Provide data-native radius-bin initialization with behavioral tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Extract/reuse useful radius-to-mass and lognormal PDF/PMF generation
    from `PresetParticleRadiusBuilder` and direct radius construction into the
    approved native construction module. Preserve documented defaults and
    units; return native data without activity/surface/distribution strategies.
  - Tests: Migrate distribution shape, radius/mass/density/charge, invalid
    distribution input, PDF versus PMF and unit-conversion assertions. Add
    independent population totals at V=0.25/4 using the M1 convention; do not
    require identical raw arrays if the approved explicit conversion differs.
  - Gate: Approved binned capabilities work natively, all enduring assertions
    are traceable, and no facade construction is needed by the native path.

- [ ] **E9-M2-P5: Provide data-native resolved initialization with non-unit-volume tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Provide direct resolved mass and sampled lognormal initialization
    using existing utilities/native builders. Preserve density/charge shape
    behavior, count/volume meaning and reviewed random-state contract; do not
    expand to new sampling physics or promise cross-backend RNG replay.
  - Tests: Migrate resolved-builder behavioral coverage; use controlled sample
    fixtures for mass conversion and independent inventory oracles at non-unit
    volumes. Test species/density/charge mismatch, invalid inputs and approved
    empty/inactive behavior. Verify partitioning and nonpartitioning gas stay
    ordered when the initialized particles enter the aggregate.
  - Gate: All inventoried retained construction capabilities have native tests;
    no double normalization, altered scientific tolerances or lost coverage.

- [ ] **E9-M2-P6: Update development documentation**
  - Issue: TBD | Size: XS | Status: Not Started
  - Work: Publish bounded construction/access/replacement developer guidance
    with M1 authority links, units/defaults and explicit no-GPU-rebind warning.
    Finalize capability, behavioral-test and temporary-compatibility ledgers.
    Broad supported tutorial/notebook migration remains M5.
  - Tests: Data-only integration from native initialization through builder,
    read access, individual/coordinated replacement and rejection snapshots;
    retain non-unit-volume/mixed-gas quantities and transfer/export smoke
    assertions. Rerun required focused groups, then untargeted coverage,
    lint/type checks and strict docs. Unit tests already ship in P1–P5.
  - Gate: Record final revision/date, literal results, skips and reviewer
    approval. M3 starts only after M2 is completed and validated; no open
    blocking construction/compatibility decision or required unavailable check.
