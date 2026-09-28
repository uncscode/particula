# Phase Details

At drafting time, all five phases were Not Started. P1 characterization for
issue #1610 is now committed (`5d8f36964`), but its scientific gate remains
pending; P2–P5 are Not Started. Execute strictly in the order below; each
is one bounded reviewable PR. Keep helper production changes near the template's
rough 100-line increment; escalate a larger discovered requirement for review
instead of pulling sibling migration work into this plan.

- [ ] **E9-M1-P1: Freeze ordered species, units and ownership contracts with characterization tests**
  - Issue: #1610 | Size: S | Status: Characterization committed; scientific sign-off pending (gate not passed)
  - Entry: E9 scope approved; no prior maintenance track required.
  - Goal: Remove semantic ambiguity before any new helper or constructor.
  - Work: Inventory facade accessors and actual native consumers by supported
    distribution on CPU and GPU. Apply E9 appendix D1–D3: shared distribution_type,
    counts/radius-PDF storage, gas-order environment and explicit process maps.
    Enumerate each helper with its consumer; freeze PDF quadrature and metadata
    construction/admission. Record old semantic conflicts as correction rows.
  - Tests: Adjacent characterization cases for existing properties/copies,
    mixed partitioning flags and non-unit-volume legacy normalization; expected
    quantities come from independent arithmetic, not the facade under test.
  - Gate: Scientific reviewer approves a complete normalization/alignment
    ledger, focused assertions pass, and P1 contract questions are resolved.

- [ ] **E9-M1-P2: Specify identity-preserving whole-container replacement and rejection gates**
  - Issue: TBD | Size: XS | Status: Not Started
  - Entry: P1 completed and validated.
  - Goal: Deliver an unambiguous aggregate acceptance contract to M2.
  - Work: Specify acceptance cases for approved properties and replace_data and
    cross-container validation order. Cover same-object assignment, compatible
    candidate identity, invalid individual replacement, valid all-three layout
    change, rejection with no partial publication, and no GPU/resident rebinding.
    This phase writes specifications only, not flat Aerosol or its setters.
  - Tests: Add or retain adjacent constructor/copy characterization assertions
    grounding the read-only-validation specification. Publish M2 acceptance
    cases without introducing expected-failing future-Aerosol tests here.
  - Gate: Maintainer approves the complete acceptance matrix and field-level
    copy/view rules; specification distinguishes structural and process checks.

- [ ] **E9-M1-P3: Implement necessary particle data helpers with non-unit-volume tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Entry: P2 completed and validated.
  - Goal: Fill only approved particle access/mutation gaps for later migration.
  - Work: Reuse existing derived properties; add minimal missing helpers with
    explicit concentration interpretation, units and read/write contracts.
    Preserve arrays as sole authority; add shared distribution metadata and
    normalization/population helpers without migrating scientific consumers.
    Audit required metadata propagation through copies and explicit Warp transfer;
    reject unsupported interpretation rather than silently relabel existing data.
  - Tests: Co-located per-particle versus population mass, radii/fractions,
    V=0.25/4 normalization, multi-species lanes, empty/inactive data, read-only
    rejection, copy/view and direct-mutation freshness assertions as applicable.
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
