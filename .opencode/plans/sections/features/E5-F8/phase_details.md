# Phase Details

- [x] **E5-F8-P1:** Create independent CPU-Warp condensation parity walkthrough
  with regression tests
  - Issue: #1367 | Size: S | Status: Implemented
  - Goal: Add a deterministic fp64 walkthrough that constructs CPU oracle and
    Warp execution inputs independently and runs the required Warp CPU path.
  - Files: `docs/Examples/gpu_condensation_parity_walkthrough.py`,
    `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py`
  - Tests: immutable/non-aliasing construction, four-substep uptake/evaporation
    oracle, no-Warp/force-disabled behavior, explicit sidecars and sync, enabled
    failure propagation, Warp CPU parity, and optional CUDA parity.

- [x] **E5-F8-P2:** Publish separate physics, conservation, and energy acceptance
  criteria with tests
  - Issue: #1368 | Size: S | Status: Implemented
  - Goal: Produce three labeled result blocks whose thresholds and failures are
    evaluated independently, including explicit no-Warp unavailability.
  - Files: `docs/Examples/gpu_condensation_parity_walkthrough.py`,
    `particula/gpu/tests/gpu_condensation_parity_walkthrough_test.py`
  - Tests: independent physics parity plus exact vapor pressure;
    per-box/per-species inventory conservation; signed finalized-transfer energy
    identity; isolated category failures and multi-failure reporting proving one
    result cannot hide another; Warp CPU and optional CUDA categorized results.

- [x] **E5-F8-P3:** Record downstream ownership for every deferred condensation
  capability with validation
  - Issue: #1369 | Size: S | Status: Implemented
  - Goal: Publish and validate the complete owner/gate/non-claim record for the
    E4 carry-forward without broadening the direct-kernel evidence boundary.
  - Files: `docs/Features/Roadmap/condensation-parity-walkthrough.md`,
    `particula/tests/condensation_parity_walkthrough_docs_test.py`
  - Tests: exactly one parseable 14-row ownership table with non-empty owners,
    gates, and non-claims; exact links and anchors; focused commands; and the
    direct-kernel-only support boundary.

- [x] **E5-F8-P4:** Update development documentation
  - Issue: #1370 | Size: XS | Status: Implemented
  - Goal: Link the walkthrough and ownership record from canonical condensation,
    foundations, example, and roadmap indexes for E5-F9 closeout.
  - Files: `docs/Features/condensation_strategy_system.md`,
    `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Examples/index.md`, `docs/Features/Roadmap/index.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`
  - Tests: the documentation regression validates exact once-only resolving
    links, focused reproduction commands, separate evidence labels, caller-owned
    write-only `energy_transfer` wording, Warp CPU/optional-CUDA policy, and
    deferred-capability non-claims across all five pages.
