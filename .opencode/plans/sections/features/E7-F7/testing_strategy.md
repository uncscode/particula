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
- **P3:** Test one-way advection, symmetric mixing, chains, arbitrary box pairs,
  physical inverse-time rates integrated over varied `time_step` values,
  edge-order permutations, closed-ledger gas conservation, and declared open
  source/sink accounting. Rejected calls must not commit gas or volume.
- **P4:** Test whole and fractional particle populations, matching and free
  destination slots, inactive slots, composition/charge preservation, multiple
  inbound edges, capacity exhaustion, deterministic selection, and no partial
  commit. Check number, per-species mass, and charge ledgers.
- **P5:** Test canonical scheduler placement, derived-state invalidation,
  resource reuse, stable identities, zero normal-step conversion/sync/readback,
  checkpoint/restart metadata, capability failures, and post-launch faulting.
- **P6:** Parameterize independent-box, 1D parcel/advection, mixing, expansion,
  and combined repeated-step fixtures. Warp CPU is required when Warp is
  installed; CUDA rows are optional and skip cleanly. Compare equivalent
  one-box references and independent NumPy ledgers with recorded `rtol`/`atol`.
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

Likely test locations are `particula/execution/tests/communication_test.py`,
`particula/gpu/kernels/tests/communication_test.py`, and
`particula/gpu/tests/communication_parity_test.py`.
