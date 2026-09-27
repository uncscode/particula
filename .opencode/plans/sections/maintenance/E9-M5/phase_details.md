# Phase Details

All seven phases are Not Started, serial and issue-sized. Every phase consumes
its predecessor's validated gate; P1 additionally requires M4 completed and
validated. Example functions and their unit/regression tests ship together.
Notebook sync/execution occurs in the owning phase, not deferred to P7.

- [ ] **E9-M5-P1: Inventory supported workflows and migrate foundational tutorials with regression fixtures**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Reconcile appendix inventory with full examples tree, indexes, API
    snippets and navigation. Record old/new API and numerical-baseline mapping.
    Migrate Aerosol, Gas_Species, AtmosphereTutorial, Particle_Representation,
    distribution/activity/surface construction and data-container foundations.
    Pure function-only pages may remain unchanged with explicit justification.
  - Tests: Add/update foundation documentation fixtures and
    `particula/gpu/tests/data_containers_example_test.py`; preserve ordered
    species, non-unit-volume normalization and held-container identity checks.
  - Gate: Inventory approved; foundation scripts and affected pairs execute,
    assertions pass and changed landing pages render strictly.

- [ ] **E9-M5-P2: Migrate condensation and coagulation examples with scientific checks**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Migrate all Dynamics/Condensation and Dynamics/Coagulation supported
    object-pattern notebooks, including latent heat, staggered, charged,
    distribution comparison and time evolution. Audit functional subdirectories
    too; retain their independent scientific reference role.
  - Tests: Co-located example regression fixtures compare species-resolved
    outcomes/conservation against retained M3 expectations; check distribution
    interpretation and non-unit volume without changing algorithms/tolerances.
  - Gate: Each inventoried pair synced/executed; deterministic and stochastic
    checks separated; process-owned configuration and single-box scope clear.

- [ ] **E9-M5-P3: Migrate wall-loss dilution and nucleation examples with behavioral checks**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Migrate Chamber_Wall_Loss notebooks, `cpu_dilution.py`,
    `Nucleation/cpu_nucleation.py` and Custom_Nucleation_Single_Species pair.
    Use M4 native held state, preserving geometry, charge and source controls.
  - Tests: Update owning docs/example fixtures including
    `particula/tests/nucleation_docs_test.py`; check all-gas dilution, current-gas
    substeps, concentration-weighted conservation, identity and failure claims.
    Run both CPU wall-loss strategy suites as assertions.
  - Gate: CPU entrypoints and pairs execute with retained scientific outcomes;
    no advertised whole-call nucleation rollback or new wall-loss capability.

- [ ] **E9-M5-P4: Migrate composed simulation notebooks with outcome regressions**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Migrate the six Simulations/Notebooks pairs and Dynamics/Customization/
    Adding_Particles_During_Simulation pair. Audit Activity and Equilibria
    families for aggregate construction and migrate affected rows here.
  - Tests: Add bounded numerical fixtures for each changed simulation family,
    preserving process order, environment history, inventories and plotted
    physical quantities. Reduced fixtures supplement full notebook execution,
    not replace it; retain stable seeds and aggregate stochastic bounds.
  - Gate: All supported simulation pairs run and retain their scientific intent;
    unsupported external resources/runtime requirements are explicit blockers,
    not justification to silently remove an example from support.

- [ ] **E9-M5-P5: Reconcile explicit-transfer and resident examples with boundary tests**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Audit/migrate all root direct-GPU and resident scripts plus direct
    nucleation. Preserve standalone versus resident workflow distinctions,
    ordered gas-name ownership, explicit transfers and concrete-only setup.
    Leave already-native scripts unchanged where verified by the inventory.
  - Tests: Run existing data-container/direct-kernel/process-sequence example
    suites and resident session/multi-timestep/graph-capture documentation
    fixtures; adapt native CPU construction only where needed. Verify no
    implicit rebinding, transfer, fallback or promotion of private APIs.
  - Gate: Installed-Warp CPU evidence passes for supported uncaptured paths;
    native CUDA-only rows pass or cleanly skip without fallback. Existing
    unavailable profiling/performance evidence remains unavailable.

- [ ] **E9-M5-P6: Publish native API migration and advanced replacement guidance with executable fixtures**
  - Issue: TBD | Size: S | Status: Not Started
  - Work: Rewrite particle-data-migration coexistence guidance for final native
    workflows, update affected API-reference inputs and topic links. Provide
    advanced field/derived access, individual and coordinated container
    replacement examples using approved M1/M2 names. Mark old snippets as
    unsupported historical before examples and keep them nonexecutable.
  - Tests: Executable after-snippet fixtures verify shapes/units, copy/view
    semantics, identity, compatible replacement, mismatched species/layout
    rejection with unchanged old/candidate state and subsequent process
    validation. Documentation guards verify history labels and supported imports.
  - Gate: Advanced examples execute, API targets resolve, strict docs build
    passes; no native capabilities invented to make documentation convenient.

- [ ] **E9-M5-P7: Update development documentation**
  - Issue: TBD | Size: XS | Status: Not Started
  - Work: Publish contributor example/pair maintenance guidance, close the
    inventory and legacy-history allowlist, and prepare the M6 handoff record.
  - Tests: Final integration audit of all supported entrypoints, snippets and
    pair outputs; run focused docs/example assertions, untargeted repository
    coverage, required lint/type checks and strict MkDocs build at final revision.
  - Gate: Literal dated command outcomes and device skips accepted by reviewer;
    zero unclassified supported examples or remaining executable legacy uses.
    Only this completed-and-validated gate authorizes M6 removal.
