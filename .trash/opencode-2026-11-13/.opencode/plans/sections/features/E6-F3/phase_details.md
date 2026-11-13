# Phase Details

## Sequencing

Complete P1 through P5 in order before P6 validates the complete step; P7
documents only the validated direct-process contract and evidence.

- [x] **E6-F3-P1:** Port and validate neutral wall-loss transport primitives with unit tests
  - Issue: #1401 | Size: S | Status: Shipped
  - Delivered: Consolidated neutral fp64 particle-transport helpers into
    `particula.gpu.properties`; removed legacy GPU-dynamics definitions and
    re-exports; migrated consumers; defined Cunningham slip zero/invalid
    behavior; and added device-only `debye_1_wp` and `x_coth_x_wp`.
  - Evidence: `particula/gpu/properties/tests/particle_properties_test.py`
    exercises transport parity, sentinel/domain behavior, Debye branch
    boundaries against an independent host oracle, and the `x_coth_x` numerical
    switch. Existing migrated dynamics, kernel, support, and benchmark consumer
    coverage verifies the new import surface.
  - Boundary: No wall-loss coefficient assembly/API, charged physics, removal
    or RNG logic, or CPU behavior change.

- [x] **E6-F3-P2:** Implement spherical and rectangular coefficient device functions with CPU parity tests
  - Issue: #1402 | Size: S | Status: Shipped
  - Delivered: Added concrete internal fp64 Warp helpers
    `spherical_wall_loss_coefficient_wp` and
    `rectangle_wall_loss_coefficient_wp`. They compose P1 transport primitives,
    use transport-input settling, and implement neutral Crump-Seinfeld
    spherical/rectangular coefficients; the rectangular form uses
    `x_coth_x_wp` for small-argument stability.
  - Files: `particula/gpu/dynamics/wall_loss_funcs.py`, `particula/gpu/dynamics/tests/wall_loss_funcs_test.py`
  - Evidence: Guarded Warp CPU/optional-CUDA smoke and CPU-oracle parity tests
    cover scalar diffusion/gravity regimes and vector nanometer-to-micrometer
    lanes. Rectangular parity uses `rtol=1e-10, atol=1e-20`; spherical parity
    records `rtol=1.002e-3` for the measured existing CPU Debye endpoint
    quadrature discrepancy.
  - Boundary: No public export, CPU change, configuration/preflight, charged
    physics, containers/particle mutation, or RNG behavior.

- [x] **E6-F3-P3:** Define neutral wall-loss configuration and atomic preflight tests
  - Issue: #1403 | Size: S | Status: Shipped
  - Delivered: Added frozen concrete-module `NeutralWallLossConfig`, write-free
    `wall_loss_step_gpu` preflight, and lazy direct-kernel export. Valid calls
    return the original particle object; all execution, coefficient, removal,
    RNG lifecycle, and output allocation remain deferred.
  - Files: `particula/gpu/kernels/wall_loss.py`,
    `particula/gpu/kernels/__init__.py`,
    `particula/gpu/kernels/tests/wall_loss_test.py`
   - Evidence: Warp-guarded tests cover configuration/import boundaries,
     direct/environment inputs, particle/RNG metadata validation, and atomicity
     snapshots; monkeypatched coefficient helpers confirm P3 never invokes them.
   - Documentation: `docs/index.md` and `AGENTS.md` record the bounded P3
     direct-kernel contract, concrete configuration import, caller ownership,
     deferred execution limits, and focused test command. Comprehensive P4-P7
     behavior documentation remains deferred.

- [x] **E6-F3-P4:** Implement fixed-shape coefficient and stochastic removal kernels with unit tests
  - Issue: #1404 | Size: S | Status: Shipped
  - Delivered: Evaluates usable-slot neutral coefficients, makes local survival
    draws, and clears mass/concentration/charge for removed fixed slots after
    frozen preflight.
  - Files: `particula/gpu/kernels/wall_loss.py`,
    `particula/gpu/kernels/tests/wall_loss_test.py`
  - Evidence: Inactive gaps, controlled survivor/removal paths, zero-time no-op,
    multi-box/species layouts, fixed-slot clearing, identity preservation, and
    pre-launch atomicity coverage.

- [x] **E6-F3-P5:** Add caller-owned persistent RNG lifecycle with repeated-step tests
  - Issue: #1405 | Size: S | Status: Shipped
  - Delivered: Replaced P4's local seed/slot draw path with caller-owned
    per-box lifecycle. Successful positive-time calls allocate and seed omitted
    private state, or reuse supplied state by identity; only
    `initialize_rng=True` resets supplied state. One sequential owner advances
    each box in ascending fixed-slot order for eligible slots only.
  - Files: `particula/gpu/kernels/wall_loss.py`,
    `particula/gpu/kernels/tests/wall_loss_test.py`, `AGENTS.md`, `readme.md`,
    `docs/index.md`, `docs/Features/data-containers-and-gpu-foundations.md`,
    `docs/Features/Roadmap/data-oriented-gpu.md`, and
    `.opencode/guides/architecture/architecture_outline.md`
  - Evidence: Focused lifecycle coverage verifies omitted-state convenience,
    initialize-once reuse, explicit reset, independent boxes, eligible-only
    consumption, all-ineligible no-draw, and exact sidecar preservation for
    zero time and rejected preflight. An opt-in benchmark-marked smoke test
    covers the sequential path without a throughput threshold.
  - Boundary: The sidecar remains external and is never returned. No hidden
    transfer/fallback, runnable, charged physics, cross-device RNG-trajectory
    parity, or GPU performance claim is added.

- [x] **E6-F3-P6:** Add deterministic coefficient and statistical CPU-Warp parity matrix
  - Issue: #1406 | Size: S | Status: Shipped
  - Delivered: Added the test-only, non-mutating Warp CPU/optional-CUDA parity
    matrix in `particula/gpu/kernels/tests/wall_loss_parity_test.py`. It compares
    complete-slot spherical/rectangular coefficient eligibility and finite rates
    with independent CPU system-state equations; spherical uses
    `rtol=1.002e-3, atol=1e-20` and rectangular `rtol=1e-10, atol=1e-20`.
  - Evidence: One-/multi-box, per-box environment, nanometer/micrometer,
    sparse/unusable-slot fixtures; 100-seed fresh-state and persistent-sidecar
    aggregate survival checks using 3-sigma binomial bounds; exact zero-time and
    all-inactive no-ops; and lazy import/configuration-boundary smoke coverage.
  - Boundary: No production physics, entry-point behavior, or public exports
    changed. CPU/Warp random-stream replay remains explicitly unsupported.

- [x] **E6-F3-P7:** Update development documentation
  - Issue: #1407 | Size: XS | Status: Shipped | Completed: 2026-07-23
  - Delivered: Published the finalized P1–P6 direct-neutral contract: SI inputs,
    explicit environment alternative, fixed-slot clearing/preservation,
    asynchronous caller ownership, persistent RNG lifecycle, evidence, and
    deferred boundaries.
  - Files: `AGENTS.md`, `docs/Features/data-containers-and-gpu-foundations.md`,
    and `docs/Features/Roadmap/data-oriented-gpu.md`.
  - Evidence: `pytest particula/gpu/dynamics/tests/wall_loss_funcs_test.py -q
    -Werror` and `pytest particula/gpu/kernels/tests/wall_loss_test.py
    particula/gpu/kernels/tests/wall_loss_parity_test.py -q -Werror`.
  - Boundary: Documentation-only; no production kernel, export, physics, or
    runtime test-matrix change. Parent [E6](../../../epics/E6.json), charged
    successor [E6-F4](../../../features/E6-F4.json), and closeout
    [E6-F9](../../../features/E6-F9.json) retain their recorded statuses.

## P4 Delivery Update (#1404)

- [x] **E6-F3-P4:** Implement fixed-shape coefficient and stochastic removal
  kernels with unit tests
  - Status: Shipped.
  - Delivered: Positive-time calls retain P3 validation ordering, normalize the
    environment only after preflight, evaluate neutral spherical/rectangular
    coefficients for usable active fixed slots, use deterministic local
    seed-plus-flattened-slot draws, and clear all mass lanes, concentration, and
    charge for removed slots. Zero time is post-preflight and write-free.
  - Files: `particula/gpu/kernels/wall_loss.py`,
    `particula/gpu/kernels/tests/wall_loss_test.py`, `docs/index.md`,
    `.opencode/guides/architecture/architecture_outline.md`, and
    `.opencode/guides/architecture/architecture_guide.md`.
  - Evidence: Focused Warp-guarded coverage includes both geometries, private
    mask survivor/clearing branches, sparse one-/multi-box and species layouts,
    zero time, controlled survival/removal, aggregate stochastic behavior, and
    pre-launch atomicity.
  - Boundary: `rng_states` remains validated but unmodified; its initialization
    and advancement are P5 work. CPU/Warp stochastic trajectory parity is P6.
