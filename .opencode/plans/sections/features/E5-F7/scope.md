# Scope

E5-F7 consolidates release-grade correctness evidence for the executable GPU
coagulation mechanisms delivered by E5-F3 through E5-F6. It adds shared fixture
and oracle infrastructure only where needed, executes the complete support
matrix on Warp CPU, conditionally reuses it on CUDA, and publishes the resulting
contract for E5-F9 closeout.

## In Scope

### Completed in issue #1362 (P1)

- `particula/gpu/kernels/tests/_coagulation_validation_support.py` supplies
  literal mask tables, explicit fp64 fixtures, and independent Brownian,
  charged, SP2016, ST1956, additive, property, and selector-majorant oracles.
- `particula/gpu/kernels/tests/coagulation_validation_test.py` supplies
  Warp-free configuration/boundary coverage and lazy Warp-CPU pair, property,
  and majorant observations for the executable matrix.
- The implementation is validation-only: it adds no production physics, APIs,
  CUDA coverage, end-to-end merge/conservation coverage, stochastic coverage,
  caller-buffer coverage, CPU fallback, or user documentation.

### Completed in issue #1363 (P2)

- `particula/gpu/kernels/tests/coagulation_validation_test.py` now runs the
  public `coagulation_step_gpu()` entry point for every executable mask on
  `normal` and heterogeneous `two_box` fixtures, each materialized with one and
  two species.
- The test-only helpers materialize explicit fp64 particle state and snapshots;
  the public assertions cover per-box/per-species inventory (`rtol=1e-12`,
  `atol=1e-30`), applicable charge conservation, legal/disjoint accepted pairs,
  donor clearing, recipient bookkeeping, and inactive sentinels.
- The completed matrix covers sparse and two-active boundaries, charged and
  sedimentation zero-rate no-ops, collision-capacity rejection, collision/count
  sidecar return identity, persistent RNG initialization/advance behavior, and
  scalar and device-array turbulent inputs for masks 8 and 10.
- Deferred masks 7, 11, 13, and 14, plus selected invalid charged,
  sedimentation, and turbulent inputs, prove exact particle/output/RNG snapshot
  preservation on preflight failure. Warp CPU is exercised when installed and
  CUDA is included only when available through `warp_devices()`.
- The implementation remains validation-only; it adds no production physics,
  API, user documentation, CPU fallback, repeated-seed statistics, or required
  CUDA evidence.

### Remaining scope

- Deterministic fp64 pair/property parity for Brownian, charged, SP2016
  sedimentation with efficiency 1, ST1956 turbulent shear, and every approved
  additive row, using independent CPU APIs or direct NumPy equations.
- Proof that each pair rate is finite and non-negative and does not exceed its
  independently checked mechanism or summed majorant.
- Bounded aggregate stochastic validation over repeated fresh seeds or steps,
  with declared confidence or sigma bounds rather than exact pair replay.
- A concise validation matrix and focused reproduction commands for E5-F9.

## Out of Scope

- New coagulation physics, mechanisms, supported combinations, production API
  redesign, or changes to the majorants established by E5-F3 through E5-F6.
- Exact CPU/Warp collision-pair or random-stream replay.
- DNS turbulence, non-unit sedimentation collision efficiency, binned or
  continuous-PDF coagulation, dynamic slot allocation, hidden CPU fallback, or
  hidden transfers.
- Performance benchmarks, throughput claims, high-level `Aerosol`/`Runnable`
  integration, graph capture, broad autodiff, or adaptive stepping.
- Making CUDA hardware mandatory for routine validation or release.
