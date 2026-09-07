# GPU resident graph capture

Native resident graph capture is concrete, direct-import-only machinery in
`particula.execution.graph_capture`; it is not a package or top-level API. The
only eligible path is a caller-qualified, non-CPU native Warp CUDA device. CPU
and Warp CPU are neither capture paths, capture evidence, nor fallback or
emulation paths. The maintained [P1 walkthrough](../Examples/gpu_resident_graph_capture.py)
owns composition code; this runbook records operations rather than providing a
second runnable recipe. Checkpoint and restart limits are defined by the
[GPU resident checkpoint contract](gpu_resident_checkpoints.md).

## Ordered setup and capture

Fail closed in this order:

1. Resolve native-CUDA capability and probes before CPU fixture construction,
   resident imports, CPU upload, or allocation. Only an unavailable capability
   or probe outcome is a clean skip: do no setup or capture and do not fall
   back. Once `qualify_prepared_resident_graph_capture()` is attempted, a
   qualification rejection raises `ValueError`; it is not a clean skip.
2. Construct the exact ACTIVE resident session, pinned resource registry, and
   closed step guard.
3. Register the complete fixed inventory; build exact views, capacities, and
   `CaptureResourceRequirements`; then publish that exact resource set once with
   `GPUResourceRegistry.prepare_capture_resources()` before final request
   construction.
4. Create and attach the exact final request, signature, lifecycle, and binding.
    Explicitly initialize the `coagulation` and `wall_loss` resident streams
    before preparation and capture. Normal replay never initializes or resets
    streams; an explicit resident-stream lifecycle operation is the only reset
    boundary.
5. Prepare the resident simulation, validate the published resource set,
    qualify the exact READY binding, and explicitly capture it to reach
    CAPTURED. Qualification is read-only and its rejected `ValueError` leaves
    the existing lifecycle available as defined by that lifecycle.

Replay requires an authentic issued record; the exact attached request, session,
registry, and closed guard; an ACTIVE session; qualified CUDA availability; a
CAPTURED lifecycle; an unchanged compatible signature; and the matching duration.
One call is one replay. Before `.numpy()` or any other host observation, call
`warp.synchronize_device(...)` explicitly.

## Structural changes require fresh capture

Compatibility is identity based and checks the following first-drift order.
Replacing any listed item fails closed and invalidates replay. Recovery is
ordered: retire the invalidated metadata, renew it to `READY`, prepare and
qualify the renewed binding, then explicitly capture before replay is possible.

| First-drift group | Representative replacement |
| --- | --- |
| `request` | Final resident request object |
| `session` | ACTIVE resident session |
| `device` | Session device declaration |
| `dimensions` | Fixed resident dimensions |
| `primary_containers` | Particle, gas, or environment container |
| `primary_arrays` | A particle, gas, or environment array |
| `resource_views` | Process view, scratch sidecar, or communication view |
| `graph` | Process graph or graph declarations |
| `schedule` | Resolved schedule or declarations |
| `schedule_order` | Ordered schedule-node identifiers |
| `diagnostics` | Diagnostic operation, registration, or output |
| `communication` | Communication node, map, buffers, or final volumes |
| `configurations` | Process configuration, capture requirements, or published set |
| `rng_resources` | Coagulation or wall-loss RNG sidecar |

Same-object payload updates, active/free-slot changes, and advancing resident
RNG words are non-triggers. They remain compatible because the tracked
identities have not changed.

## Lifecycle and incidents

| State | Replay | Next action |
| --- | --- | --- |
| `READY` | No | Prepare, qualify, and explicitly capture. |
| `CAPTURED` | Yes, after every replay precondition | Replay, or replace an identity to invalidate before retirement. |
| `INVALIDATED` | No | Retire stale metadata, renew it to `READY`, then prepare, qualify, and explicitly capture. |
| `FAULTED` | No | Do not retry; close and create fresh setup as needed. |
| `RETIRED` | No | Renew the retired binding to `READY`, then prepare, qualify, and explicitly capture. |
| `CLOSED` | No | Terminal; create a fresh session and binding. |

Structural drift invalidates capture. A writer-capable capture or replay failure
faults state and promises no rollback or retry. A read-only qualification or
admission rejection leaves the existing state available as defined by its
lifecycle. Lifecycle ownership releases every opaque native handle exactly once;
a release failure does not restore provenance or trigger retry, rollback, or
recovery. Close graph capture before closing the session; terminal session,
close, discard, or finalize changes stale the capture binding.

## Retirement, renewal, and restart

Retire stale or invalidated metadata. Renew only a retired binding. Renewal
creates READY metadata only: it is neither replay nor recapture. Re-prepare,
re-qualify, and explicitly capture after renewal.

## Concrete operations

Import these operations directly from `particula.execution.graph_capture`; none
is a package or top-level export. Resolve capability with
`resolve_graph_capture_capability()`, then qualify a prepared `READY` binding
with `qualify_prepared_resident_graph_capture()`. Capture the qualification with
`capture_prepared_resident_graph()` and replay its authentic issued record with
`replay_captured_resident_graph()`. For an invalidated resident binding, call
`retire_resident_graph_capture()`, then
`renew_resident_graph_capture()` to produce its fresh `READY` lifecycle. Use
`close_resident_graph_capture()` for ordered teardown before closing the
resident session.

Checkpoint restart and terminal closure require a fresh session, resource and
array identities, qualification, setup, and capture. Restart restores resident
bytes and RNG continuation only; it creates fresh identities and no graph
handles.

## Boundaries and prohibitions

Native handles are opaque, nonserializable, and uncheckpointed. There is no
stale reuse, migration, serialization, cross-device replay, CPU or Warp-CPU
capture/emulation/fallback, hidden transfer/readback/synchronization, automatic
recapture/retry/rollback, resizing, compaction, dynamic shapes, dynamic order,
dynamic maps, or portable performance claim.

## Reproduction commands

Run focused assertions before the untargeted coverage runner. The CUDA-selected
row is optional pass-or-clean-skip evidence only; it is never CPU or Warp-CPU
substitution.

```bash
pytest particula/tests/gpu_graph_capture_runbook_docs_test.py \
  particula/tests/gpu_resident_graph_capture_docs_test.py -q --no-cov
pytest particula/execution/tests/graph_capture_test.py -q --no-cov
pytest particula/execution/tests/captured_full_loop_test.py -q --no-cov
pytest particula/execution/tests/captured_full_loop_test.py -q \
  -m "warp and cuda" --no-cov
.opencode/tools/run_pytest.py
mkdocs build --strict
```

Relevant repository sources are
`particula/execution/tests/graph_capture_test.py`,
`particula/execution/tests/captured_full_loop_test.py`,
`particula/tests/gpu_graph_capture_runbook_docs_test.py`, and
`particula/tests/gpu_resident_graph_capture_docs_test.py`.
