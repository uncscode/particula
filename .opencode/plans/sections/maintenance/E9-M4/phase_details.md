# Phase Details

All five phases are Not Started. Each is a bounded reviewable PR. P1 requires
M3 completed and validated; every later phase requires its predecessor's
validated gate. Unit tests ship with the function changes, not in a later phase.

- [ ] **E9-M4-P1: Migrate scientific runnable state access and composition with behavioral tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Accept M1–M3 contracts and establish the consumer/assertion ledger.
    Migrate `MassCondensation`, `Coagulation` and `WallLoss` rate/execute paths
    in `particle_process.py` to held data and environment. Preserve native
    strategy identities and correct clipping without duplicate normalization.
    Retain `runnable.py` sequence semantics; change only necessary native seams.
  - Tests: Migrate `particula/tests/runnable_test.py` and
    `dynamics/tests/wall_loss_runnable_test.py`; add adjacent scientific-runnable
    assertions as needed. Verify call order/durations, read-only rate results,
    identity, environment forwarding, protected gas fields, non-unit volume,
    current-state visibility, rejection and per-substep clipping.
  - Gate: Native runnable assertions and applicable M3 scientific regressions
    pass without facade construction; no new scientific algorithm or exports.

- [ ] **E9-M4-P2: Migrate dilution state access and atomic updates with all-gas tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Migrate `dilution.py` physical-rate, preflight, candidate, commit and
    restore paths to unified gas and native particle storage. Update `Dilution`
    only where required; preserve concrete preflight and custom delegation.
  - Tests: Migrate `dynamics/tests/dilution_test.py` and
    `dilution_runnable_test.py`; retain `dilution_exports_test.py`. Exercise
    mixed/all-false/all-true partitioning masks, scalar validation, non-unit
    volumes, zero-work validation, malformed storage, invalid candidate,
    identity and unchanged protected fields. Inject failure after an earlier
    write to prove restoration and original-exception propagation; retain
    rollback-failure chaining and custom strategy boundary tests.
  - Gate: All gas lanes follow independent `exp(-alpha * dt)` expectations;
    rejection and failure contracts remain intact without legacy setters.

- [ ] **E9-M4-P3: Migrate nucleation orchestration with current-gas and substep failure tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Replace `_topology` facade admission with approved native single-box
    topology/mapping. Read held environment and gas; stop synchronizing facade
    caches on the native path. Resolve the explicit-environment argument and
    distribution admission questions before code changes. Reuse P2/P3 source
    finalization/commit primitives and unchanged exhaustion policy.
  - Tests: Migrate `dynamics/tests/nucleation_runnable_test.py` with
    `dynamics/nucleation/tests/particle_source_test.py` regression checks.
    Retain rate/depletion, precursor, validity-domain, identity, zero time,
    substep bounds, no-admission, exhaustion and weighted conservation cases.
    Inject a later attempted-substep failure: prior commits remain, the failed
    attempt leaves its entry state unchanged. Cover mixed species mapping and
    direct/environment replacement without stale cached state.
  - Gate: Native nucleation uses current gas each equal substep and retains
    per-attempted-substep atomicity; no whole-call rollback or physics expansion.

- [ ] **E9-M4-P4: Migrate CPU adapter integration with dispatch and ownership tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Migrate CPU consumers/fixtures in `execution/adapters/condensation.py`
    and `coagulation.py` to native aggregates and M3 strategies. Avoid gratuitous
    adapter edits when identity-preserving delegation already works. Preserve
    existing isothermal condensation/Brownian coagulation profile boundaries.
  - Tests: Migrate `condensation_adapter_test.py`, `coagulation_adapter_test.py`,
    `condensation_integration_test.py` and `coagulation_integration_test.py` in
    `execution/tests/`. Retain capability rejection, configuration/runnable
    identity, original payload/result identity, exact call counts, equal
    substeps and errors/no-retry. Verify CPU replacement does not touch prepared
    GPU bindings, and retain fallback-policy/export tests unchanged in intent.
  - Gate: Direct CPU and selected-adapter native workflows agree under existing
    tolerances; no transfers, fallback, GPU rebind or capability expansion.

- [ ] **E9-M4-P5: Update development documentation**
  - Issue: TBD | Size: XS | Status: Not Started
  - Work: Publish narrow developer call/ownership/failure guidance and finalized
    consumer/assertion/compatibility ledgers for M5/M6. Broad supported-example
    and notebook migration remains M5-owned; final legacy deletion stays M6.
  - Tests: Add final native composed integration checks in
    `particula/integration_tests/`, including mixed gas, non-unit volume and
    failure propagation. Rerun focused groups, both CPU wall-loss suites,
    retained transfer/export/ownership checks, then untargeted repository
    coverage and required lint/type checks; strict docs build for changed docs.
  - Gate: Record final revision/date, literal command outcomes and skips;
    reviewer accepts all M4 criteria and authorizes M5. Required unavailable
    evidence or unresolved contract questions block that authorization.
