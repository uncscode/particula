# Overview

- **Problem Statement:** E6-F7 establishes an inventory-limited CPU nucleation
  reference, but device-resident simulations still lack a direct Warp operation
  that turns the same bounded rates into fixed-shape particle sources. Moving
  state to the host would violate ownership, transfer, and composition goals.
- **Value Proposition:** A low-level GPU step can finalize gas-feasible demand,
  consume E6-F5 slot discovery and E6-F6 exhaustion policy, and commit matching
  gas depletion and particle mass without resizing arrays, hidden transfers, or
  CPU fallback. Independent parity and conservation evidence makes that path
  scientifically auditable.
- **User Stories:**
  - As a GPU simulation author, I want nucleation to operate on caller-owned
    Warp state so repeated process sequences require no intermediate host copy.
  - As a scientific user, I want the direct step to match the E6-F7 CPU oracle
    and conserve every box/species inventory so GPU results remain trustworthy.
  - As a library maintainer, I want fixed-shape sidecars and fail-before-write
   validation so invalid calls cannot partially mutate simulation state.

## Delivered: P1 (#1438), P2 (#1439), P3 (#1440), P4 (#1441), and P5 (#1442)

`particula/gpu/kernels/nucleation.py` now provides the concrete-only, read-only
P1 boundary with frozen configuration and caller-owned sidecar dataclasses plus
private validation/preflight. Co-located Warp tests cover its ownership,
schema, validation-order, and no-write gate/rejection contract. It deliberately
does not expose or execute a GPU nucleation step, allocate fallback storage,
transfer state, or mutate caller particle or gas data.

Private `_plan_nucleation_demand(...)` now implements P2. It reuses P1
preflight to calculate survival-included activation or kinetic `J`,
`E_pot = J * dt`, and one inventory-limited accepted demand per box. Its sole
commit writes P2-owned demand, removal, and gate-diagnostic sidecars; it leaves
 P3 request buffers, particles, and `gas.concentration` unchanged.

Private `_stage_nucleation_slots(...)` now implements P3 (#1440). After P2
admission and reused P1 preflight, it converts exact finite nonnegative
`accepted_demand * volume` products to retained `wp.int32` provisional counts,
then reuses E6-F5 `get_slot_diagnostics_gpu` for active/free layouts. It writes
only caller-owned P3 sidecars: full accepted counts, E6-F5 diagnostics, and the
free-slot selectable prefix with `-1` tails. Counts may exceed free capacity;
P4 alone resolves that capacity policy and activates slots. Conversion and E6-F5
preflight failures preserve those sidecars; rollback is not promised after an
asynchronous diagnostic or P3 commit launch.

Private `_orchestrate_nucleation_exhaustion(...)` now implements P4 (#1441).
It preserves P2 accepted demand and P3 count/diagnostic records as immutable
handoffs, chooses fully viable E6-F6 resampling before scaling fallback, and
writes P4 final demand/count/free-prefix diagnostics. Expected all-box
rejections occur before P4 workspace writes or primitive entry and preserve
particle, gas, P2, P3, P4, and nested scratch state. Once a primitive is
entered, its documented no-rollback boundary applies; P4 claims no
cross-primitive rollback. P4 adds no public API, E6-F9 integration, or direct
 activation; selected E6-F6 primitives may mutate documented particle fields,
 while P4 does not mutate gas or source mass.

Issue #1442 delivers P5 as the supported, lazily exported
`particula.gpu.kernels.nucleation_step_gpu(...)` boundary. It composes P1--P4,
validates the finalized P4 handoff, and launches one fused device commit that
initializes only selected fixed-capacity slots and removes the corresponding
finalized gas inventory. It returns the identical particle and gas containers.
Precommit rejections preserve particle/gas state; P2--P4 sidecars retain their
phase-owned mutation boundaries, and entered E6-F6 primitive limits still
apply. The direct same-device path adds no transfer, CPU fallback, resize,
compaction, Runnable, or E6-F9 integration.
