# Overview

**Problem Statement:** Particula has backend-selected condensation and Brownian
coagulation plus a resident GPU session, but no production scheduler that orders
all supported processes and state refreshes. Ad hoc loops can consume stale
temperature-dependent vapor pressure or saturation state, move gas at the wrong
time, reset stochastic resources, or violate the session's no-transfer contract.

**Value Proposition:** E7-F5 turns the shipped adapters and direct-process
boundaries into one validated, deterministic timestep. It executes supported
condensation, coagulation, dilution, wall loss, and nucleation against resident
state, applies environment and gas updates in dependency order, exposes bounded
diagnostic hooks, and performs no implicit conversion, synchronization, or
fallback. This preserves issue #1451 Track T5 and unblocks E7-F7, E7-F8, and
E7-F9.

**User Stories:**

- As a simulation user, I want the same declared process set to run in a stable
  order every timestep so results do not depend on registration order.
- As a GPU user, I want environment, derived thermodynamic, particle, gas,
  sidecar, and RNG state to remain resident between checkpoints.
- As a maintainer, I want invalid graphs and stale-state hazards rejected before
  launch so process physics remains delegated to its owning implementation.

Parent epic: E7. Scope authority: issue #1451, Track T5.

## Delivered: E7-F5-P1 (#1492)

`particula/execution/process_graph.py` now provides the unexported, pure
backend-neutral declaration and validation boundary. It validates immutable
closed-catalogue nodes and allowed dependency edges, detects cycles, and
normalizes node and edge declaration order deterministically. It deliberately
does not schedule or execute work, access resident resources, select a backend,
or change `particula.execution` exports. Later phases retain scheduler and
resident-timestep delivery.

## Delivered: E7-F5-P2 (#1493)

`resolve_canonical_topological_order()` now provides lexical Kahn ordering for
validated process-graph declarations. The new direct-import-only
`particula.execution.scheduler` resolves immutable enabled-node selections and
direction profiles into declaration-only schedules, retaining dependency closure
and deriving freshness edges before topology resolution. It has no launch,
lifecycle, resource, backend, or export side effects; resident timestep
execution remains later work.

## Delivered: E7-F5-P4 (#1495)

`particula.execution.state_updates` now supplies a concrete-only, direct-import
boundary for prescribed resident environment and gas replacements bound to the
resolved `environment_update` and `gas_update` graph nodes. Immutable requests
retain exact resident session, registry, graph, node, and caller-owned Warp
payload identities. Ordered preflight validates that binding, schemas, device,
contiguity, nonempty alias ranges, and physical values before in-place copies;
canonical empty schemas are write-free no-ops. The boundary neither schedules
nodes nor refreshes derived state, transfers host data, provides fallback or
exports, or changes resident lifecycle behavior.
