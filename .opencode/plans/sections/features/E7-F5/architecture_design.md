# Architecture Design

## High-Level Design

### P1 implemented declaration boundary

`process_graph.py` is below the future orchestration layer and above no runtime
resource layer: it imports only standard-library modules plus E7-F1 execution
declarations and condensation capability metadata. `resolve_timestep_plan()`
validates a supplied immutable `TimestepPlan` against a private closed
catalogue, allowed dependency pairs, and acyclicity, then returns a new
`ResolvedProcessGraph` whose nodes and edges are sorted by identifier. The
resolved record deliberately has no topological/execution-order field.

The catalogue uses E7-F1's condensation matrix for condensation requirements;
the other supported process rows use explicitly empty baseline requirements.
This P1 layer neither invokes `begin_step()` nor imports resource views, Warp,
or GPU modules. Scheduling, hazard insertion, disabled-node handling, and
execution remain P2 and later work.

The scheduler is a typed orchestration layer above process adapters and E7-F4
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
direction is universal. Disabled nodes disappear only after dependency
validation.

## Data / API / Workflow Changes

- **Data Model:** P1 added immutable `ProcessNode`, `NodeKind`,
  `DependencyEdge`, `TimestepPlan`, and `ResolvedProcessGraph` declarations.
  They reference E7-F1 capability IDs only; E7-F4 process-resource views,
  update declarations, and hooks remain later work. CPU/Warp schemas do not
  change.
- **API Surface:** Add a scheduler entry point such as
  `SimulationScheduler.step(session, plan, time_step)` through the deliberate
  `particula.execution` export policy. Concrete adapters and scratch records
  remain module-owned.
- **Workflow Hooks:** Pre-step validation is read-only. `begin_step()` opens the
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
