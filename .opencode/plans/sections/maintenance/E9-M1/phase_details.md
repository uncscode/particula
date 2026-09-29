# Phase Details

At drafting time, all five phases were Not Started. P1 characterization for
issue #1610 was committed (`5d8f36964`); P1 scientific and P2 specification
approvals are recorded in `appendix.md`. P3 is merged, P4 validation is in
progress, and P5 remains Not Started. Execute strictly in the order below; each
is one bounded reviewable PR. Keep helper production changes near the template's
rough 100-line increment; escalate a larger discovered requirement for review
instead of pulling sibling migration work into this plan.

Maintainer representation/grid/volume/builder clarifications and explicit P1
scientific/P2 specification approval from issue #1612 are recorded in this
track's `appendix.md`; they supersede conflicting raw-count/raw-PDF proposals.
Historical characterization does not become revised-contract test evidence by
approval. P3 implementation and independent tests are present in #1612's
merged workflow; the P3 completion reconciliation is recorded below.

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
  - Issue: #1612 | Size: S | Status: Implemented; completed and merged in workflow state 2026-09-29
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
  - Issue: #1613 | Size: S | Status: P4 fix implemented; final gate pending
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

## P4 fix handoff (issue #1613, 2026-09-29)

- P1 scientific and P2 specification approvals are recorded in `appendix.md`;
  these approvals do not supply implementation evidence. P3 predecessor gate
  reconciled from issue #1612 workflow `8ef4d823`: `fix_completed=true`,
  the post-fix Validate, Polish, Run Tests, Format and Ship Auto Implementation
  phases are completed, and `branch_merged=true` (`current_phase` reports
  "Workflow completed with skipped steps"; earlier pre-fix Validate/Polish/
  Test/Format phases remain pending, replaced by the completed post-fix phases).
  Worktree HEAD `0fd3472be` includes the inherited #1612 P3 commits. This
  is a workflow-state and commit reconciliation, not a fresh P3 science review.
- The direct-only P4 checker and adjacent regression tests are in
  `particula/aerosol_validation.py` and
  `particula/tests/aerosol_validation_test.py`. It does not certify same-width
  ratio chemistry, transform masks, or modify inherited P3 code.
- The checker accepts `ParticleData` masses `(B,N,Sp)`, concentration and
  charge `(B,N)`, density `(Sp,)`, volume `(B,)`; `GasData` molar mass and
  partitioning `(Sg,)`, concentration `(B,Sg)`, nonblank unique ordered names;
  `EnvironmentData` temperature and pressure `(B,)` and saturation ratio
  `(B,Sg)`. Only `B>0` and `Sg>0` are required: `Sp` may differ from `Sg`,
  empty particle capacity and all-false masks pass. Rejection is read-only;
  callers may correct metadata and retry without rollback or partial state.
  Later M3/M4 admission must compare full ordered expected configuration names
  with current gas names and verify authoritative ratio-producer order when
  available (otherwise alignment is unverified), require unique in-range
  one-to-one integer `(gas_index, particle_index)` pairs, classify unmatched
  lanes, check mapped mask eligibility, physical-domain and distribution
  capability before mutation. This checker neither implements those gates nor
  detects same-width ratio permutations; an all-false mask admits no transfer.
- Focused CPU gate on the fix revision: `pytest` for gas_data_test.py,
  environment_data_test.py and aerosol_validation_test.py with `-q`, coverage
  disabled: **111 passed**. Focused GPU conversion/export gate with `-q`,
  coverage disabled: **150 passed**; existing transfer tests already assert
  mixed/all-false masks, every gas/ratio lane, names and detached restores,
  so no conversion production change was needed. Lint, mypy, untargeted
  coverage and strict MkDocs gates are not inferred from focused results;
  their final results are recorded separately below.
- Final style-only revision test delegation: **31 focused passed; 6,975 full
  suite passed, 20 skipped; 93% full-package coverage (80% threshold)**.
  Documentation delegation: **strict MkDocs passed** with link and formatting
  checks. On the subsequent P4 lint gate, Ruff check and format check passed
  on `particula/` (520 files); mypy `particula/ --ignore-missing-imports`
  passed (520 source files). The lint agent corrected only the two new P4
  Python files. After those corrections, focused CPU **111 passed** and GPU
  transfer/export **150 passed**; delegated full-suite **6,975 passed, 20
  skipped, 93% coverage**; strict MkDocs **passed** (links and formatting
  checked). M1/P5 is not declared shipped by this P4 gate.
- Validator follow-up at worktree HEAD `0fd3472be` plus the pending P4 diff:
  focused gas/environment/checker assertions **139 passed** (coverage disabled),
  GPU conversion/exports **150 passed** (coverage disabled), Ruff check and
  format-check **passed**, mypy on `particula/` **passed** (520 files), and
  untargeted repository-policy coverage **7,003 passed, 20 skipped, 93%**
  (80% threshold). The additional constructor-input nonmutation tests and
  one-pass ordered-name inspection are included in this evidence. The strict
  MkDocs validation-only equivalent returned exit code 0; it was not the
  literal raw `mkdocs build --strict` invocation. These are local worktree
  results, not a claim that M1/P5 has shipped.
