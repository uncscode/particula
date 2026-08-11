# Phase Details

- [x] **E7-F9-P1:** Add GPU diagnostics reductions and co-located contract tests
  - Issue: #1528 | Size: S | Status: Completed 2026-08-10
  - Delivered: Concrete-only six-operation resident diagnostics protocol: two
    preserved snapshots plus total species mass, particle-number concentration,
    latent-heat energy, and conservation residual reductions without normal-step
    host readback or synchronization.
  - Files: `particula/execution/diagnostics.py`,
    `particula/execution/gpu_resources.py`,
    `particula/execution/tests/diagnostics_test.py`, and
    `particula/execution/tests/gpu_resources_test.py`.
  - Tests: Contract coverage for reduction equations, identity, canonical order,
    empty shapes, invalid schema/capacity/alias preflight, and pre-launch
    no-write boundaries. Public docs and exports remain unchanged.

- [x] **E7-F9-P2:** Freeze checkpoint schema and add round-trip validation tests
  - Issue: #1529 | Size: S | Status: Completed 2026-08-10
  - Delivered: Froze schema-v3 required continuation metadata while allowing an
    empty current-word payload. Clarified checkpoint docstrings so canonical
    primaries and registry-owned sidecars, ledgers, diagnostics, and closed-map
    state are recovery authority, excluding arbitrary caller-owned outputs.
  - Files: `particula/execution/checkpoint.py`,
     `particula/execution/tests/checkpoint_test.py`
  - Tests: Bidirectional coagulation/wall-loss resource-continuation pairing
    rejection, canonical-primary immutability, and schema-v2 noncommunication
    restart. Broader versioned round-trip and uninterrupted-equivalence coverage
    remains follow-up evidence.

- [x] **E7-F9-P3:** Add resident full-loop transfer and ordering regressions
  - Issue: #1530 | Size: S | Status: Completed 2026-08-10
  - Delivered: Corrected the private resident-scheduler nucleation branch to
    execute `ResidentNucleationAdapter` and then record ordinary completion.
    Nucleation is not a thermodynamic consumer; `_CONSUMER_IDS` and the
    canonical twelve-node order remain unchanged.
  - Files: `particula/execution/resident_scheduler.py` and
    `particula/execution/tests/full_loop_test.py`.
  - Tests: Added repeated real resident-loop rows for closed GAS and PARTICLES
    maps, exact ordinary-node traces, refresh windows and NumPy derived-state
    observations, one-upload/identity-schema assertions, and tight closed GAS
    inventory conservation (`rtol=1e-12`, `atol=1e-30`). Added a late wall-loss
    writer-failure regression proving token closure, `FAULTED` lifecycle, and
    later-dispatch rejection without asserting rollback.

- [x] **E7-F9-P4:** Add independent multi-box parity and isolation regressions
  - Issue: #1531 | Size: S | Status: Completed 2026-08-10
  - Delivered: Added only `particula/execution/tests/multi_box_loop_test.py`.
    The internal regressions retain real resident lifecycle/scheduler boundaries
    while covering logical-ID rather than lane-based equivalence, independent
    multi-box decomposition, added/reordered/no-work isolation, resident stream
    ownership, and neutral wall-loss aggregate behavior.
  - Files: `particula/execution/tests/multi_box_loop_test.py`
  - Tests: 4-box, 16-slot, 2-species fixed-capacity zero-duration parity and
    tight closed inventory; selected/empty wall-loss and all-free no-work stream
    preservation; positive-duration same-backend stream continuity; 100-seed
    Warp-CPU binomial aggregate evidence; and optional 12-seed CUDA bounded
    smoke coverage. No production code, exports, checkpoints, or public docs
    changed.

- [x] **E7-F9-P5:** Add transport expansion conservation and restart regressions
  - Issue: #1532 | Size: S | Status: Completed 2026-08-11
  - Delivered: Test-only regression coverage; production APIs, checkpoint schema,
    scheduler ordering, exports, and user documentation are unchanged.
  - Files: `particula/execution/tests/transport_loop_test.py`,
    `particula/execution/tests/restart_loop_test.py`, and
    `particula/gpu/kernels/tests/communication_test.py`.
  - Tests: Independent NumPy float64 extensive-amount oracle checks for two-step
    directed expansion/reciprocal mixing with dilution, sparse closed-map
    conservation, and empty/disabled write-free barriers; exact-device restart
    with fresh restored identities and preserved published stream words; and
    direct open-boundary source-minus-sink ledger reconciliation at
    `rtol=1e-12`, `atol=1e-30`.

- [x] **E7-F9-P6:** Publish complete multi-timestep example with documentation regression
  - Issue: #1533 | Size: S | Status: Completed 2026-08-11
  - Delivered: Published a runnable three-box resident-scheduler example with
    lazy no-Warp guidance, availability validation, one source setup upload, two
    source dispatches, caller-owned diagnostics, manual exact-device
    checkpoint/restart, and cached source finalization. It uses concrete resident
    seams and no direct process-kernel orchestration or CPU fallback.
  - Files: `docs/Examples/gpu_resident_multi_timestep.py`,
     `particula/tests/gpu_resident_multi_timestep_docs_test.py`
  - Tests: Documentation regression covers forced-disabled, missing-Warp, broken
    enabled-import, availability, setup, and dispatch-failure behavior, plus real
    Warp-CPU enabled execution when available. It asserts one source upload per
    CPU container, resident identity preservation, diagnostic shapes, manual
    restart/finalization semantics, and warning-clean subprocess behavior.

- [x] **E7-F9-P7:** Publish support contract validation matrix and closeout evidence
  - Issue: #1534 | Size: XS | Status: Completed 2026-08-11
  - Goal: Update user/developer documentation, record tolerances and commands,
    run the required matrix, and publish dated evidence for every Epic G exit item.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, `.opencode/guides/`
  - Evidence: Focused 289 assertions, exports 15, resident suite 891,
    full-package coverage 93%, execution scope 95% (P1--P6 aggregate 86%), and
    optional CUDA rows 1 and 5 all passed. Exact `mkdocs build --strict` also
    passed, so P7 and Epic G are shipped.
