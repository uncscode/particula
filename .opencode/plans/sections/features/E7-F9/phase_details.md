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

- [ ] **E7-F9-P2:** Freeze checkpoint schema and add round-trip validation tests
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Audit and freeze the minimum versioned payload needed to restore all
    documented physical, semantic, lifecycle, configuration, and RNG state.
  - Files: `particula/execution/checkpoint.py`,
    `particula/execution/tests/checkpoint_test.py`
  - Tests: Schema/version rejection, ordered metadata, exact same-backend stream
    continuation, uninterrupted equivalence, and malformed payload safety.

- [ ] **E7-F9-P3:** Add resident full-loop transfer and ordering regressions
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Exercise all supported processes over repeated steps while proving one
    setup upload, stable identities, current derived state, and no transfer or
    synchronization before an explicit checkpoint.
  - Files: `particula/execution/tests/full_loop_test.py`
  - Tests: Independent CPU/Warp CPU results, exact call order, transfer spies,
    shape/identity stability, conservation, errors, and fault propagation.

- [ ] **E7-F9-P4:** Add independent multi-box parity and isolation regressions
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Compare a larger particle-resolved multi-box loop with equivalent
    one-box references and prove unrelated/disabled boxes remain isolated.
  - Files: `particula/execution/tests/multi_box_loop_test.py`
  - Tests: One-box decomposition, reordered/disabled/added-box metamorphic rows,
    deterministic fields, stochastic statistics/stream contracts, and capacity.

- [ ] **E7-F9-P5:** Add transport expansion conservation and restart regressions
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Validate prescribed advection, dilution, mixing, and volume evolution
    plus checkpoint/restart against independent CPU extensive-amount oracles.
  - Files: `particula/execution/tests/transport_loop_test.py`,
    `particula/execution/tests/restart_loop_test.py`
  - Tests: Closed conservation, open ledgers, sparse/disconnected maps, expansion,
    no overdraw, same-backend restart, and persistent per-box RNG.

- [ ] **E7-F9-P6:** Publish complete multi-timestep example with documentation regression
  - Issue: TBD | Size: S | Status: Not Started
  - Goal: Publish one user-facing loop using backend selection, all supported
    process categories, multi-box state, diagnostics, and explicit checkpoints.
  - Files: `docs/Examples/gpu_resident_multi_timestep.py`,
    `particula/tests/gpu_resident_multi_timestep_docs_test.py`
  - Tests: Execute on Warp CPU when available, validate import behavior without
    Warp, assert checkpoint-only restores, and treat warnings as errors.

- [ ] **E7-F9-P7:** Publish support contract validation matrix and closeout evidence
  - Issue: TBD | Size: XS | Status: Not Started
  - Goal: Update user/developer documentation, record tolerances and commands,
    run the required matrix, and publish dated evidence for every Epic G exit item.
  - Files: `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, `AGENTS.md`, `.opencode/guides/`
  - Tests: Focused and full fast suites, coverage >=80% for changed modules,
    optional CUDA rows, export checks, and `mkdocs build --strict`.
