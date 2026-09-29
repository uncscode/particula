# Phase Details

At drafting time, all five phases were Not Started. P1 characterization for
issue #1610 is now committed (`5d8f36964`), but its scientific gate remains
pending; P2–P5 are Not Started. Execute strictly in the order below; each
is one bounded reviewable PR. Keep helper production changes near the template's
rough 100-line increment; escalate a larger discovered requirement for review
instead of pulling sibling migration work into this plan.

Maintainer representation/grid/volume/builder clarifications and explicit P1
scientific/P2 specification approval from issue #1612 are recorded in this
track's `appendix.md`; they supersede conflicting raw-count/raw-PDF proposals.
Historical characterization does not become revised-contract test evidence by
approval. P3 implementation and independent tests are present in the #1612
worktree; final review/ship evidence remains pending.

- [ ] **E9-M1-P1: Freeze ordered species, units and ownership contracts with characterization tests**
  - Issue: #1610 | Size: S | Status: Characterization committed; revised scientific contract approved 2026-09-28; historical tests remain characterization
  - Entry: E9 scope approved; no prior maintenance track required.
  - Goal: Remove semantic ambiguity before any new helper or constructor.
  - Work: Inventory facade accessors and actual native consumers by supported
    distribution on CPU and GPU. Reconcile E9 appendix D1–D3 with the later
    maintainer clarification: already-per-volume PMF/PDF at fixed V=1,
    represented resolved counts at physical V, explicit radius-PDF grid and
    trapezoids, gas-order environment and explicit process maps.
    Enumerate each helper with its consumer; freeze PDF quadrature and metadata
    construction/admission. Record old semantic conflicts as correction rows.
  - Tests: Adjacent characterization cases for existing properties/copies,
    mixed partitioning flags and non-unit-volume legacy normalization; expected
    quantities come from independent arithmetic, not the facade under test.
    Retain legacy behavior tests as characterization; replace PDF V≠1 oracles
    with fixed-V=1 m⁻⁴ PDF integrals (varying species masses and zero PDF on a
    valid grid). Use V=.25/1/4 only for explicitly resolved-count density;
    add PMF bin-sum density and specify builder invalid-kind/volume/grid cases;
    runnable new-helper and builder rejection tests follow their implementation
    in P3, not as expected-failing tests in this specification-only phase.
  - Gate: Scientific reviewer approves a complete normalization/alignment
    ledger, focused assertions pass, and P1 contract questions are resolved.

- [ ] **E9-M1-P2: Specify identity-preserving whole-container replacement and rejection gates**
  - Issue: TBD | Size: XS | Status: Acceptance/ownership specification approved 2026-09-28; aggregate implementation deferred to M2
  - Entry: P1 completed and validated.
  - Goal: Deliver an unambiguous aggregate acceptance contract to M2.
  - Work: Specify acceptance cases for approved properties and replace_data and
    cross-container validation order. Cover same-object assignment, compatible
    candidate identity, invalid individual replacement, valid all-three layout
    change, rejection with no partial publication, and no GPU/resident rebinding.
    Use the proposed matrix in `appendix.md` as a review checklist, not an
    already-approved outcome. Specify that candidate metadata and writable
    input receive read-only representation validation; aggregate box, gas-name
    and environment width checks remain structural; distribution capability,
    lane mapping and single-box physics belong to process admission. Freeze
    field-level copy/view rules and validation order for one-bad-box, empty,
    same-object and all-three replacements, preserving old and candidate state
    on rejection. No implicit PMF/PDF volume rescaling or GPU rebinding.
    This phase writes specifications only, not flat Aerosol or its setters.
  - Tests: Add or retain adjacent constructor/copy characterization assertions
    grounding the read-only-validation specification. Publish M2 acceptance
    cases without introducing expected-failing future-Aerosol tests here.
  - Gate: First require a revised P1 ledger and independent unit/grid evidence
    at a recorded revision plus Kyle's explicit P1 scientific approval. Then
    require reviewed P2 acceptance matrix, field-level copy/view and rejection
    rules, adjacent current-behavior characterization, recorded validation
    evidence and Kyle's explicit P2 approval before P3. Neither approval is
    inferred from this plan update.

- [ ] **E9-M1-P3: Implement necessary particle data helpers with non-unit-volume tests**
  - Issue: #1612 | Size: S | Status: Implemented in workflow worktree; final review/ship pending
  - Entry: P2 completed and validated.
  - Goal: Fill only approved particle access/mutation gaps for later migration.
  - Work: Reuse existing derived properties; add minimal missing helpers with
    explicit concentration interpretation, units and read/write contracts.
    Preserve arrays as sole authority; add shared distribution metadata and
    representation-aware normalization/population helpers without migrating
    scientific consumers. Builders require explicit kind and validate fixed
    PMF/PDF V=1 or positive finite resolved V on creation; do not add a blanket
    constructor check. Add a read-only `ParticleData` representation-validation
    method for direct construction and after caller mutation; new bulk helpers
    and process admission using the interpretation call the same validation.
    Writable raw-array edits are not intercepted and no automatic repair or
    conversion of mismatched representations is provided.
    Audit required metadata propagation through copies and explicit Warp transfer;
    reject unsupported interpretation rather than silently relabel existing data.
  - Tests: Co-located per-particle versus population mass, radii/fractions,
    V=0.25/4 resolved normalization, fixed-V=1 PMF and PDF, multi-species
    lanes, empty/inactive data, read-only rejection after direct construction
    and mutation without mutation by validator, copy/view and direct-mutation
    freshness assertions as applicable.
  - Gate: All changed helpers have independent oracles, no double normalization,
    approved metadata only, unchanged physics ownership, and passing tests/lint.

- [ ] **E9-M1-P4: Implement necessary gas and environment helpers with alignment tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Entry: P3 completed and validated.
  - Goal: Make ordered mixed-gas and environment access safe for later consumers.
  - Work: Implement only P1-approved missing helpers or reusable read-only
    alignment validation. Preserve names, molar masses, partitioning flags and
    concentrations in one order; document candidate mutation/copy behavior.
    No aggregate publication, dilution loop or adapter migration is included.
  - Tests: Co-located interleaved partitioning/nonpartitioning fixtures,
    all-false mask, mismatched dimensions/configuration order, duplicate/missing
    names according to the approved policy, failed-update nonmutation and
    physical domain tests. Retain supported CPU↔Warp conversion assertions.
  - Gate: No species dropped/reordered, no hidden mutation during validation,
    and focused tests/lint pass for every changed function.

- [ ] **E9-M1-P5: Update development documentation**
  - Issue: TBD | Size: XS | Status: Not Started
  - Entry: P4 completed and validated.
  - Goal: Publish the developer contract and verified handoff, not broad M5 docs.
  - Work: Update the bounded container reference, add a data-only integration
    example/test of the approved helpers if needed, and finalize decision,
    helper and validation ledgers. Distinguish delivered helpers from specified
    future aggregate APIs. Reconcile cross-container units/order/copy examples.
  - Tests: Data-only integration regression with mixed gas and non-unit volume;
    rerun focused groups, untargeted repository runner, lint/mypy and strict
    documentation checks. Unit tests already ship in their owning phases.
  - Gate: Approved P1/P2 decisions plus literal successful required evidence at
    the final revision authorize M2. Required failures/unavailable checks keep
    M1 incomplete. No parallel M2–M6 implementation is authorized beforehand.
