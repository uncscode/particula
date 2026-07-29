# Architecture Design

## High-Level Design

### P1/P2 implemented declaration and scheduling boundary

`process_graph.py` is below the future orchestration layer and above no runtime
resource layer: it imports only standard-library modules plus E7-F1 execution
declarations and condensation capability metadata. `resolve_timestep_plan()`
validates a supplied immutable `TimestepPlan` against a private closed
catalogue, allowed dependency pairs, and acyclicity, then returns a new
`ResolvedProcessGraph` whose nodes and edges are sorted by identifier. The
resolved record deliberately has no topological/execution-order field.

P2 adds the concrete, read-only
`resolve_canonical_topological_order()` helper to `process_graph.py`. It uses
lexically ordered Kahn traversal over validated nodes and edges, rejects missing
endpoints and effective cycles, and still does not add an order field to
`ResolvedProcessGraph`.

The catalogue uses E7-F1's condensation matrix for condensation requirements;
the other supported process rows use explicitly empty baseline requirements.
This P1 layer neither invokes `begin_step()` nor imports resource views, Warp,
or GPU modules. Runtime scheduling and execution remain later work.

P2's `particula.execution.scheduler` is direct-import-only and declaration-only.
It calls P1 resolution before selection or profile work, then applies enabled
selection, explicit-predecessor closure, the reviewed nucleation/condensation
direction, and required freshness dependencies before delegating canonical order
to the P1 helper. It imports no resident lifecycle/resource, adapter, GPU, or
Warp code and has no launch, mutation, transfer, synchronization, or fallback
behavior.

### P4 implemented resident state-update boundary

`particula.execution.state_updates` is a concrete-only, direct-import boundary
below any future runtime scheduler. Frozen `eq=False` environment and gas
request carriers retain exact session, pinned registry, resolved graph, node,
and caller-owned Warp update arrays. Its executor first validates the pinned
session, exact graph-node membership and canonical update role, then validates
target/input schema, resident device, contiguity, identity/nonempty byte-range
nonaliasing, and scalar payloads. Only successful preflight performs in-place
`wp.copy` commits: temperature then pressure, or gas concentration.

The executor leaves particle data (including authoritative volume), gas molar
mass/partitioning/vapor pressure, environment saturation ratio, sidecars, graph
declarations, and lifecycle state untouched. Empty canonical box or zero-species
schemas skip scalar scans and writers. It does not execute a graph, establish
freshness or derived-state refresh, acquire resources, transfer data to host,
 provide fallback, or make lifecycle guarantees after a launched copy failure.

### P5 implemented thermodynamic-freshness boundary

`particula.execution.thermodynamic_updates` is a concrete-only, direct-import
Warp boundary, not an extension of the declaration-only scheduler. Its frozen
request retains one exact active `ResidentSession`, its pinned
`GPUResourceRegistry`, a resolver-produced `ResolvedProcessGraph`, a resolved
schedule, and `ThermodynamicsConfig`. Before each operation it validates the
binding, graph provenance, exact schedule-member identities and dependencies,
and canonical thermodynamic node roles.

`ResidentThermodynamicUpdateCoordinator` begins with vapor pressure and
saturation ratio stale. `record_completed()` is caller-owned reporting after a
successful ordinary node; it records only that node's declared invalidations
and advances a local cursor. `execute_consumer()` accepts only the next
canonical condensation or diagnostics node and a zero-argument callback. It
consumes the immediately preceding virtual `vapor_pressure_refresh` then
`saturation_refresh` nodes, writes only stale fields, and invokes the callback
once. A successful consumer records its invalidations; notably condensation
makes saturation stale for a later consumer.

Vapor pressure delegates exclusively to
`particula.gpu.kernels.thermodynamics.refresh_vapor_pressure_gpu()`. A private
Warp kernel writes saturation as `concentration * GAS_CONSTANT * temperature /
(molar_mass * vapor_pressure)` on resident `(n_boxes, n_species)` lanes. The
coordinator itself performs no host payload read, synchronization, conversion,
fallback, or CPU vapor-pressure calculation; delegated configuration
fingerprint reads retain the primitive's existing behavior. A failed virtual
writer or callback leaves the cursor unchanged. If vapor pressure succeeds and
saturation fails, only vapor pressure becomes fresh; rollback after a writer
launch is not promised. The boundary neither owns lifecycle/guard tokens nor
acquires resources, dispatches a whole timestep or general process, transfers,
restores gas, or changes exports.

### P6 implemented diagnostics and runtime scheduler

`particula.execution.diagnostics` is a concrete-only, direct-import closed
protocol. Its only operations snapshot current resident gas concentration or
saturation ratio into separately caller-owned `(B, S)` outputs. It binds plans
and registrations by identity to the active session, pinned registry,
resolver-produced graph/schedule, and final diagnostics node; registry
validation rejects aliases with primaries, published sidecars, or other outputs.
Canonical empty schemas are valid write-free no-dispatch no-ops. It exposes no
callback, readback, transfer, synchronization, checkpoint, or package export.

`particula.execution.resident_scheduler` is the typed orchestration layer above
existing process boundaries and E7-F4 lifecycle. It accepts only the exact
resolver-produced ten-node schedule and exact request/node/session/registry/
closed-guard bindings. It validates the complete composition and stored-duration
agreement before `begin_step()`, then opens exactly one token and dispatches
`schedule.ordered_node_ids` without substituting a handwritten order. It does
not implement process physics or infer fallback from an exception.

```text
TimestepPlan + ExecutionContext + ResidentSession
                 |
          validate capabilities, dimensions, resources, graph
                 |
                 v
             begin_step()
                 |
 environment update -> gas update -> virtual refreshes -> condensation
                  |
 brownian coagulation -> dilution -> wall loss -> nucleation -> diagnostics
                 |
            complete_step()
```

The exact graph, rather than input list order, is authoritative. Required edges
ensure environment changes precede derived-state refresh and every consumer sees
current gas and thermodynamic state. Independent nodes use stable process IDs as
the tie breaker. Every reviewed scheduling profile declares exactly one fixed
edge between nucleation and condensation for its configured workflow; neither
direction is universal. P2 selection occurs only after complete P1 validation;
an enabled consumer cannot retain a missing explicit or derived predecessor.
Virtual refresh IDs are consumed only through the condensation and diagnostics
consumer windows, never independently dispatched. A failure after a
writer-capable call closes the token, faults the session, and preserves the
operational error without a rollback guarantee; a pre-writer failure aborts the
token and leaves the session active.

## Data / API / Workflow Changes

- **Data Model:** P1 added immutable `ProcessNode`, `NodeKind`,
  `DependencyEdge`, `TimestepPlan`, and `ResolvedProcessGraph` declarations.
  They reference E7-F1 capability IDs only; E7-F4 process-resource views,
  update declarations, and hooks remain later work. CPU/Warp schemas do not
  change.
- **Data Model:** P2 adds immutable `EnabledNodeSelection`, `SchedulerProfile`,
  `NucleationCondensationDirection`, and `ResolvedTimestepSchedule` records in
  the concrete scheduler module. Schedule nodes and edges are canonically
  sorted, and the ordered IDs are a permutation of the selected nodes.
- **API Surface:** `resolve_timestep_schedule()` and the P2 records are
  direct imports from `particula.execution.scheduler`; no
  `particula.execution` or top-level export changed. The concrete runtime
  boundary is `ResidentSimulationScheduler` in the separate direct-import-only
  `particula.execution.resident_scheduler` module.
- **Workflow Hooks (P6 runtime scheduler):** Pre-step validation is read-only. `begin_step()` opens the
  mutation window; successful completion advances session time/step once.
  Prelaunch errors leave the session active and unchanged. Any exception after a
  process/update launch preserves the original error and faults the session;
  rollback is not promised.
- **State Authority:** During a GPU session, Warp particle, gas, environment,
  sidecar, diagnostic, and RNG arrays remain authoritative and identity-stable.
  Normal steps do not call conversion helpers, `wp.synchronize()`, `.numpy()`,
  checkpoint, or finalize.
- **Updates:** Prescribed update nodes accept only validated fixed-shape state or
  immutable update descriptions. Temperature and pressure remain positive and
  finite; concentrations/saturation remain finite and nonnegative. Temperature
  changes invalidate vapor pressure and saturation. Condensation's coupled gas
   mutation is retained rather than duplicated by the scheduler.
- **P4 Update API:** `ResidentEnvironmentUpdateRequest`,
  `ResidentGasUpdateRequest`, and `ResidentStateUpdateExecutor` are direct
  imports from `particula.execution.state_updates`; `particula.execution.__all__`
  remains unchanged. This is an explicit node-bound mutation seam, not a
  scheduler or derived-state boundary.
- **Diagnostics:** `ResidentDiagnosticOperation`, registrations, and plans are
  direct imports from `particula.execution.diagnostics`; only the two closed
  snapshots exist. Host callbacks/readbacks are not part of the normal GPU step.
- **Runtime API:** `ResidentSimulationRequest` and
  `ResidentSimulationScheduler` are direct imports from
  `particula.execution.resident_scheduler`; package exports remain unchanged.

## Security & Compliance

No network or credential behavior is added. Reject unknown node names, dynamic
imports, cycles, incompatible devices/shapes/dtypes, forbidden aliases, invalid
physical values, and unavailable capabilities before mutation. Resource demand
is bounded by validated fixed dimensions. Never deserialize callables or resolve
process types from untrusted strings. Explicit transition policy remains E7-F6.
