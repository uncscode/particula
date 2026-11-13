# Scope

E7-F5 adds a deterministic scheduler in `particula.execution` over the E7-F2,
E7-F3, and E7-F4 contracts. A typed process graph validates supported nodes and
dependencies, then runs one canonical resident timestep with explicit
environment, derived-thermodynamic, gas, and diagnostic boundaries.

## In Scope

- Typed capability nodes and immutable timestep/process declarations.
- Deterministic dependency resolution independent of user registration order.
- Backend-selected condensation and Brownian coagulation from E7-F2/E7-F3.
- Resident direct adapters for shipped dilution, neutral/charged wall loss, and
  fixed-slot nucleation without changing their kernel contracts.
- Concrete-only prescribed per-box temperature, pressure, and gas replacements
  at exact resolved graph nodes, with strict shape, dtype, device, contiguity,
  nonempty alias-range, finiteness, positivity/nonnegativity, and identity
  validation; canonical empty schemas are write-free no-ops.
- Vapor-pressure and saturation refresh after relevant updates and before every
  consumer; simulation volume remains `ParticleData.volume` state.
- E7-F4 `begin_step()`/`complete_step()` lifecycle integration, stable identity,
  post-launch faulting, and no intermediate transfer or synchronization.
- Optional non-mutating diagnostic hooks and complete-loop Warp CPU tests;
  optional CUDA rows skip cleanly.

## Delivered in P6 (#1497)

- Added direct-import-only `particula/execution/diagnostics.py` with the closed
  gas-concentration and saturation-ratio snapshot protocol, caller-owned output
  validation/nonaliasing, and canonical empty write-free no-ops.
- Added direct-import-only `particula/execution/resident_scheduler.py` with the
  exact complete-ten-node schedule requirement, identity-bound active
  session/registry/closed-guard lifecycle, read-only whole-loop preflight,
  one-token resolved-order dispatch, and thermodynamic consumer windows.
- Added complete-loop coverage in `particula/execution/tests/scheduler_test.py`
  and updated the concrete architecture guides. Normal scheduling performs no
  transfer, restore, explicit synchronization, checkpoint/finalize, fallback,
  resource acquisition/replacement, resize, or compaction.

## Delivered in P1 (#1492)

- Added `particula/execution/process_graph.py` with frozen node, dependency,
  plan, and resolved-graph declarations; a ten-row closed node catalogue; and
  a closed allowed-edge schema.
- Added pure validation and deterministic declaration normalization, including
  exact immutable-container checks, catalogue/requirement validation, and
  canonical cycle reporting.
- Added focused tests in `particula/execution/tests/process_graph_test.py`.
- Kept the layer unexported and backend-neutral: no Warp/GPU import, resident
  resource access, scheduler, execution order, state update, or mutation.

## Delivered in P2 (#1493)

- Added `resolve_canonical_topological_order()` to
  `particula/execution/process_graph.py`: a read-only lexical Kahn ordering
  helper with endpoint and effective-cycle rejection.
- Added direct-import-only `particula/execution/scheduler.py` with immutable
  selection, nucleation/condensation direction-profile, and resolved-schedule
  records.
- Added P1-first declaration-only schedule resolution: selected-node and
  predecessor closure, reviewed direction handling, derived freshness edges,
  deterministic effective-edge ordering, and canonical topology delegation.
- Added co-located `process_graph_test.py` and `scheduler_test.py` coverage.
- Kept package exports, lifecycle/resource behavior, process launches, backend
  imports, transfers, synchronization, and state mutation unchanged.

## Delivered in P4 (#1495)

- Added direct-import-only `particula/execution/state_updates.py` with frozen
  environment and gas update requests and a resident in-place copy executor.
- Bound requests by exact identity to the active resident session, pinned
  registry, resolved graph, and canonical `environment_update` or `gas_update`
  node before payload validation or writer launch.
- Added deterministic metadata, protected-array alias, and scalar-value
  preflight; temperature/pressure require finite positive values and gas
  concentration requires finite nonnegative values.
- Preserved target/container identities and protected particle, gas, and
  environment fields; empty-box and zero-species schemas complete without scan
  or copy.
- Added focused lazy-Warp coverage in
  `particula/execution/tests/state_updates_test.py`.
- Did not add scheduler execution, derived-state refresh, transport, host
  transfer, fallback, lifecycle behavior, or package exports.

## Delivered in P5 (#1496)

- Added direct-import-only `particula/execution/thermodynamic_updates.py` with
  exact session/registry/graph/schedule/configuration identity binding and
  resolver/schedule/role validation before every report or consumer bracket.
- Added `ResidentThermodynamicUpdateCoordinator`, seeded with stale vapor
  pressure and saturation markers. Callers report only successfully completed
  ordinary nodes; the coordinator records their declared invalidations and
  advances its local cursor.
- Added consumer bracketing for only canonical `condensation` and `diagnostics`:
  immediately preceding virtual refresh nodes are consumed in vapor-pressure
  then saturation order, and only stale fields are written.
- Reused `refresh_vapor_pressure_gpu()` as the authoritative vapor writer and
  added a private on-device saturation writer using `GAS_CONSTANT` and current
  resident concentration, molar mass, temperature, and vapor pressure.
- Preserved container and primary-array identities. Failed writers/callbacks do
  not advance the cursor; a completed vapor writer remains fresh if a following
  saturation writer fails.
- Added focused Warp tests in
  `particula/execution/tests/thermodynamic_updates_test.py` and concrete-only
  architecture-guide documentation.
- Did not add lifecycle/guard behavior, resource acquisition, scheduler or
  general-process dispatch, transfer, synchronization, fallback, gas restore,
  or package exports.

## Out of Scope

- Silent CPU fallback, hidden CPU/GPU movement, or runtime retry on another
  backend; E7-F6 owns explicit transition and error policy.
- Multi-box transport, mixing, advection, and expansion (E7-F7).
- Final per-box stream identity and restart policy (E7-F8).
- Epic-wide diagnostics products, complete public example, and closeout matrix
  (E7-F9), beyond the hooks and tests needed by this feature.
- Unsupported physics expansion, GPU staggered condensation, dynamic resizing
  or compaction, multi-GPU/distributed execution, graph capture, performance
  claims, autodiff, or kernel-physics rewrites.
