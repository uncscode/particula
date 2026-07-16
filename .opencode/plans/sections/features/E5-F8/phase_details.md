# Phase Details

- [ ] **E5-F8-P1:** Create independent CPU-Warp condensation parity walkthrough
  with regression tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Add a deterministic fp64 walkthrough that constructs CPU oracle and
    Warp execution inputs independently and runs the required Warp CPU path.
  - Files: `docs/Examples/gpu_condensation_parity_walkthrough.py`,
    `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py`
  - Tests: deterministic no-Warp behavior, independent input construction,
    Warp CPU execution, explicit synchronization, and optional CUDA skip.

- [ ] **E5-F8-P2:** Publish separate physics, conservation, and energy acceptance
  criteria with tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Produce three labeled result blocks whose thresholds and failures are
    evaluated independently.
  - Files: `docs/Examples/gpu_condensation_parity_walkthrough.py`,
    `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py`
  - Tests: particle/gas physics comparisons; per-box/per-species inventory
    conservation; signed finalized-transfer energy identity; category-isolation
    tests proving one passing category cannot satisfy another.

- [ ] **E5-F8-P3:** Record downstream ownership for every deferred condensation
  capability with validation
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Publish one complete owner/gate/non-claim table for the E4
    carry-forward and validate that required capabilities cannot disappear.
  - Files: `docs/Features/Roadmap/condensation-parity-walkthrough.md`,
    `particula/tests/condensation_parity_walkthrough_docs_test.py`
  - Tests: required row coverage, unique non-empty owner and gate fields,
    roadmap anchors, focused commands, and support-boundary wording.

- [ ] **E5-F8-P4:** Update development documentation
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Link the walkthrough and ownership record from canonical condensation,
    foundations, example, and roadmap indexes for E5-F9 closeout.
  - Files: `docs/Features/condensation_strategy_system.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Examples/index.md`, `docs/Features/Roadmap/index.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`
  - Tests: documentation link/content validation and warning-clean focused
    reproduction commands.
