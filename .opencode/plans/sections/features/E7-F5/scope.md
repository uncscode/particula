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
