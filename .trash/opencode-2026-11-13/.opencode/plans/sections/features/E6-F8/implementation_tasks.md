# Implementation Tasks

## GPU Backend

- [x] Define the bounded config, scratch, diagnostic, and request sidecars in
  `particula/gpu/kernels/nucleation.py`, documenting every dtype and shape
  (P1, #1438). The frozen records retain caller-owned arrays by identity.
- [x] Implement metadata, alias, same-device, scientific-domain, count, and
  read-only device-value preflight before clearing or allocating outputs (P1,
  #1438). P1 performs no output clearing or allocation.
- [x] Port E6-F7 activation/kinetic rate equations and SI conversions to private
  Warp P2 work storage without broadening their validity domain or model aliases
  (#1439). Survival is included once in `J`; P2 writes no particle or gas state.
- [x] Implement potential events and common per-box inventory admission, storing
  accepted demand, species-encoded limiter/gate diagnostics, and planned
  precursor removal in P2-owned sidecars (#1439). Nonparticipants retain exact
  zero removal and P3 request buffers remain untouched.
- [x] Adapt provisional demand to E6-F5 fixed-shape diagnostics rather than
  reimplementing active/free predicates (#1440). P3 performs private exact
  demand-volume int32 conversion, retains full counts, and writes the bounded
  selected free-slot prefix without activation or capacity resolution.
- [x] Privately orchestrate E6-F6 exhaustion with fully viable resampling-first
  precedence and scaling fallback (#1441). P4 preserves immutable P2/P3
  handoffs, finalizes demand/count/free-prefix diagnostics from separate
  workspace, and rejects invalid residual/final domains without truncation.
- [x] Preflight P4/nested sidecars and expected plan failures before P4 writes or
  primitive entry (#1441), preserving complete caller snapshots; document that
  entered E6-F6 primitives retain their own no-rollback boundary.
- [x] Implement one fused P5 commit that initializes finalized selected slots and
   subtracts exact finalized gas mass only after all boxes are feasible (#1442).
- [x] Add `nucleation_step_gpu(...)` with stable return/identity behavior and a
   lazy export in `particula/gpu/kernels/__init__.py` (#1442).
- [x] Guard the direct step against conversion helpers, `.numpy()` physics,
   CPU fallback, dynamic resize, and implicit synchronization (#1442).

## Tooling / Tests

- [x] Add P1 fast config/preflight validation, identity, no-op, and
  failure-immutability tests in `particula/gpu/kernels/tests/nucleation_test.py`
  (#1438). Rate, finalization, capacity, and commit coverage remain deferred.
- [x] Add P2 co-located independent float64-oracle coverage for rates, common
  admission, limiter ties, gates, inventory safety, and sidecar-only mutation
  in `particula/gpu/kernels/tests/nucleation_test.py` (#1439).
- [x] Add P3 co-located layout, conversion, sidecar ownership, and preservation
  coverage in `particula/gpu/kernels/tests/nucleation_test.py` (#1440).
- [x] Add P4 co-located policy-oracle, final diagnostic, identity, failure
  snapshot, and entered-primitive-boundary coverage in
   `particula/gpu/kernels/tests/nucleation_test.py` (#1441).
- [x] Add P5 co-located commit, handoff-validation, atomicity, direct-input,
  sidecar-identity, and no-hidden-transfer coverage, plus lazy-export tests
  (#1442).
- [x] Add 718-line `nucleation_parity_test.py` with independent NumPy float64
  P2/P3/P4/P5 expectations, one/many boxes and species, sparse/full slots,
  scaling/resampling, and repeated current-gas calls (#1443).
- [x] Assert per-box/species represented particle-plus-gas conservation rather
  than aggregate conservation alone; separately account for scaling's particle
  inventory and P5 gas transfer (#1443).
- [x] Require Warp CPU when Warp is installed and make CUDA parametrization skip
  cleanly when unavailable; retain at least 80% changed-code coverage (#1443).
- [x] Add regression assertions for exact write-free paths, explicit transfer
   boundaries, stable shapes/dtypes/devices/identities, and no mutation after
   direct-entry preflight rejection (#1443).

## Documentation / Publication

- [x] Publish the direct-Warp P1--P5 contract in feature, roadmap, theory,
  architecture, and `AGENTS.md` documentation; retain E6-F9 as the downstream
  explicit-transfer integration boundary (#1444).
- [x] Add the explicit CPU-to-Warp transfer and synchronization example at
  `docs/Examples/Nucleation/gpu_direct_nucleation.py`, without a CPU fallback
  or concrete-record package export (#1444).
- [x] Add documentation assertions and a Warp-guarded in-process/subprocess
  example regression in `particula/tests/nucleation_docs_test.py` and
  `particula/gpu/tests/gpu_direct_nucleation_example_test.py` (#1444).
