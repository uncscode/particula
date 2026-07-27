# Architecture Design

## High-Level Design

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
  nucleation -> condensation -> coagulation -> wall loss -> dilution
                 |
        gas update nodes / diagnostic hooks at declared barriers
                 |
            complete_step()
```

The exact graph, rather than input list order, is authoritative. Required edges
ensure environment changes precede derived-state refresh and every consumer sees
current gas and thermodynamic state. Independent nodes use stable process IDs as
the tie breaker. Disabled nodes disappear only after dependency validation.

## Data / API / Workflow Changes

- **Data Model:** Add immutable `ProcessNode`, `NodeKind`, `TimestepPlan`,
  `ResolvedProcessGraph`, update declarations, and hook records. They reference
  E7-F1 capability IDs and E7-F4 process-resource views; CPU/Warp container
  schemas do not change.
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
