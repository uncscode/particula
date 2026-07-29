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

### Future runtime scheduler

The runtime scheduler will be a typed orchestration layer above process adapters and E7-F4
resident-session lifecycle. It validates the complete declaration before
`begin_step()`, derives a canonical graph order, and then delegates each node.
It does not implement process physics or infer fallback from an exception.

```text
TimestepPlan + ExecutionContext + ResidentSession
                 |
          validate capabilities, dimensions, resources, graph
                 |
                 v
             begin_step()
                 |
      environment update (if declared)
                 |
      vapor-pressure + saturation refresh
                 |
  profile-declared nucleation/condensation edge -> remaining process graph
                 |
        gas update nodes / diagnostic hooks at declared barriers
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
  `particula.execution` or top-level export changed. A runtime
  `SimulationScheduler.step()` remains later work.
- **Workflow Hooks (future runtime scheduler):** Pre-step validation is read-only. `begin_step()` opens the
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
- **Diagnostics:** Hooks are ordered barriers with typed resident views and may
  write only registered diagnostic buffers. Host callbacks/readbacks are not
  part of the normal GPU step.

## Security & Compliance

No network or credential behavior is added. Reject unknown node names, dynamic
imports, cycles, incompatible devices/shapes/dtypes, forbidden aliases, invalid
physical values, and unavailable capabilities before mutation. Resource demand
is bounded by validated fixed dimensions. Never deserialize callables or resolve
process types from untrusted strings. Explicit transition policy remains E7-F6.
