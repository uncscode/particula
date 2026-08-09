# Testing Strategy

Every implementation phase includes tests in the same change. Coverage
thresholds are never lowered, test files use `*_test.py`, and changed modules
must maintain at least 80% coverage.

## Per-Phase Coverage

- **P1 (shipped, #1507):**
  `particula/execution/tests/communication_test.py` covers immutable
  declarations, valid fixed maps (including empty/all-disabled forms), ordered
  schema/device/alias/domain/topology/duplicate rejection, and write-free
  preflight identity retention. It also records the deliberate P3 handoff:
  P1 receives neither resident population nor `time_step` and does not validate
  population-dependent outbound overdraw.
- **P2 (shipped, #1508):**
  `particula/gpu/kernels/tests/communication_test.py` covers the direct
  device-resident volume operation with expansion/compression, mixed box
  factors, unchanged-volume write-free behavior, identity/protected-state
  preservation, particle/gas extensive-inventory invariants, and rejection of
  invalid schemas, aliases, domains, overflow, and underflow before mutation.
  The operation remains direct-kernel-only; it does not claim transport or
  scheduler/session behavior.
- **P3 (shipped, #1509):** The co-located
  `particula/gpu/kernels/tests/communication_test.py` tests an independent
  immutable-ledger oracle, fan-in edge-order independence, closed and open
  boundary accounting, zero-time/all-disabled write-free no-ops, aggregate
  overdraw and invalid metadata gating, resource schemas, and invalid-time
  rejection before container access. It also verifies that P3 does not mutate
  particle fields, volume, or read-only map/final-volume metadata.
- **P4 (shipped, #1510):** The co-located
  `particula/gpu/kernels/tests/communication_test.py` covers direct
  fixed-capacity particle transport. Coverage includes immutable pre-step
  planning, exact population matching and free-slot reservation, caller-owned
  ledgers and deterministic assignments, closed-map conservation, and gated
  commit behavior. The contract remains concrete-only, with Warp CPU evidence
  and optional CUDA rows where available.
- **P5 (shipped, #1511):** `particula/execution/tests/gpu_resources_test.py`,
  `checkpoint_test.py`, `process_graph_test.py`, `thermodynamic_updates_test.py`,
  `resident_communication_test.py`, and `scheduler_test.py` cover GAS/PARTICLES
  resource pinning and nonaliasing, canonical twelve-node barrier ordering,
  saturation-only invalidation, identity-stable no-op dispatch, schema-v2
  checkpoint restoration and schema-v1 noncommunication restart compatibility,
   no normal-step conversion/readback/synchronization, writer-fault guard/session
   transitions, and concrete-only export boundaries.
- **P6 (shipped, #1512):**
  `particula/gpu/tests/communication_parity_test.py` and
  `particula/execution/tests/multi_box_communication_test.py` provide test-only
  independent NumPy `float64` parity and conservation evidence for direct
  communication primitives and the concrete resident executor. They cover
  isolated/one-box-equivalent, padded 1-D, mixing, arbitrary closed,
  sparse-particle, expansion/compression, complete open work/accounting ledgers
  (direct-only), edge-permutation, and short repeated-sequence cases. Resident
  particle results use a test-local immutable-prestate NumPy planner. Warp CPU
  is the installed-Warp baseline; CUDA rows are optional and skip cleanly.
  Assertions use `rtol=1e-12` and documented `atol` declarations and keep
  parity, conservation, and open accounting separate.
- **P7:** Run strict documentation build and any contract/example regressions.

## Invariants and Tolerances

- Closed communication conserves total gas amount, particle number,
  concentration-weighted particle species mass, and concentration-weighted
  charge. Use tight `float64` tolerances justified per test; target
  `rtol=1e-12` with scale-appropriate `atol` unless kernel reduction order
  requires a documented alternative.
- Expansion/compression alone conserves extensive inventories while changing
  concentrations by `old_volume / new_volume`.
- Combined transport and volume updates construct every edge amount from
  pre-step volume, apply final volume afterward, and normalize exactly once.
- Results are independent of edge registration order and unrelated isolated
  boxes. Adding or reordering disabled boxes must not change enabled-box results.
- Empty/disabled communication and unchanged volume are write-free after full
  validation.
- No normal resident step calls conversion helpers, `.numpy()`, checkpoint,
  finalize, or `wp.synchronize()`.
- Precommit rejection preserves particle, gas, environment, volume, and
  caller-owned map arrays. Scratch mutation during documented planning is
  asserted separately. Rollback after commit launch is not promised.

P5 focused validation ran:

```bash
pytest particula/execution/tests/gpu_resources_test.py \
  particula/execution/tests/checkpoint_test.py \
  particula/execution/tests/process_graph_test.py \
  particula/execution/tests/thermodynamic_updates_test.py \
  particula/execution/tests/resident_communication_test.py \
  particula/execution/tests/scheduler_test.py -q -Werror
pytest particula/execution/tests/exports_test.py -q -Werror
pytest particula/execution/tests/ --cov=particula.execution \
  --cov-report=term-missing --cov-fail-under=80 -q -Werror
ruff check particula/execution/
ruff format particula/execution/
ruff check particula/execution/
mypy particula/execution/ --ignore-missing-imports
mkdocs build --strict
```
