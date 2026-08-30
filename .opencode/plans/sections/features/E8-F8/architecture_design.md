# Architecture Design

## High-Level Design

E8-F8 is a tested publication and evidence layer over the concrete-only graph
capture lifecycle shipped upstream. The runnable example calls production
boundaries directly and the runbook describes the same fail-closed state
machine. Closeout consumes immutable test/benchmark/profile artifacts; it does
not mutate runtime policy.

```text
ACTIVE resident binding + prepared plan + pinned resources + qualified CUDA
                              |
                              v
example: validate -> initialize RNG -> capture -> replay N times
                              |
                 +------------+-------------+
                 | compatible                | structural/lifecycle drift
                 v                           v
              replay                    reject + invalidate
                                              |
                                 explicit teardown/setup/recapture

E8-F1..F7 implementation and evidence
              -> dated closeout manifest
              -> required commands run sequentially with literal output
              -> all required rows pass? -> roadmap/parent status Shipped
                                   no  -> remain Active with blocker recorded
```

The recapture table is derived from upstream compatibility signatures. Mutable
payload values, active-slot occupancy, concentrations, and advancing RNG words
remain in pinned arrays and do not alone require recapture. Device, dimensions,
array or sidecar identity, request/schedule/process configuration,
communication map/buffers, diagnostics binding, checkpoint/restart-created
identities, explicit teardown, or terminal session/capture lifecycle do.

## Data / API / Workflow Changes

- **Data model:** No production schema change is expected. The closeout report
  uses a versioned documentation schema containing date, repository revision,
  Python/Warp/runtime/driver versions, qualified devices, command, target list,
  result, literal output/artifact link, and required/optional disposition.
- **API surface:** No package or top-level exports. The example imports
  concrete resident and graph-capture seams directly and preserves opaque graph
  handles in process memory only.
- **Workflow hooks:** P1 and P2 turn E8-F1--E8-F4 contracts into executable
  guidance. P3 consumes E8-F5--E8-F7 artifacts and runs closeout commands. P4
  updates the Epic H roadmap and E8 plan only after P3 passes.
- **Failure policy:** Known unavailable CUDA/capture capability may cleanly skip
  optional hardware rows. Missing or failed required rows, missing literal
  output, stale artifacts, or incomplete executable-target coverage block
  closeout; no result is inferred from another command.

## Security & Compliance

No network, credential, permission, or serialization boundary changes. Opaque
native handles and device pointers must never be printed, persisted, restored,
or copied into evidence. Reports may include nonsecret device/software identity
and normalized measurements. Examples and runbooks must not hide transfer,
synchronization, fallback, recapture, retry, or rollback behavior.
