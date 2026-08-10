# GPU resident checkpoints

Current controllers create schema-v3 checkpoints; v1 noncommunication and v2
communication checkpoints remain restart-compatible. Resident communication is a concrete-only
closed-map barrier: communication runs before optional prescribed volume
evolution, and both invalidate saturation ratio only.

The barrier resources and executor are direct imports under
`particula.execution`; they are not package or top-level exports. Schema-v2
restart creates fresh communication arrays and bindings rather than reusing
source identities.

The direct-import-only checkpoint boundary is available from
`particula.execution.checkpoint`; it is deliberately not exported by
`particula.execution` or the top-level package. Create a
`ResidentCheckpointController` for one active `ResidentSession`, its pinned
`GPUResourceRegistry`, and its `ResidentStepGuard`, or use the session's
`checkpoint(registry, guard)` and `finalize(registry, guard)` methods.

`checkpoint()` is nonterminal and returns a fresh immutable host snapshot.
`finalize()` is terminal and idempotent only when called again with the exact
bound session, pinned registry, and closed guard. After its first successful
call the session is `FINALIZED` and later calls with that matching binding return
the cached snapshot without device work; mismatched or invalid bindings are
rejected. Checkpoints are explicit in-memory, same-device recovery only. They
do not serialize to disk, select or migrate a device, synchronize implicitly
during restart, or provide rollback after a device writer has launched.

The snapshot owns immutable canonical bytes for primary arrays and acquired
sidecars, plus detached CPU inspection carriers. Inspection `GasData` is
intentionally lossy because CPU gas carriers do not contain GPU vapor pressure;
restart uses canonical bytes and restores vapor pressure exactly. Snapshotting
requires approximately one additional host copy of resident payload bytes plus
the detached inspection copies. Restart explicitly requires the compatible
target `Device` through `restart_resident_session(checkpoint, device)`.
`restart_checkpoint` is an equivalent concrete-only alias; both require the
same exact compatible device and create fresh session, registry, guard, and
resident-array identities.

Restart compatibility is intentionally exact and fail-closed. It accepts
`ResidentCheckpoint` records with carrier type `"ResidentSession"`, lifecycle
`ACTIVE`, complete valid canonical payload descriptors, and an exactly equal
target `Device`. Schema-v1 checkpoints must be noncommunication checkpoints.
Schema-v2 checkpoints may contain no communication family or exactly one
complete closed-map GAS or PARTICLES communication family with matching
metadata and payloads. Schema-v3 also permits absent or complete continuation
for canonical published coagulation and wall-loss streams. Its immutable current
`uint32` words are restart authority; normal dispatch and reacquisition neither
read them back nor reset them, and only explicit stream reset derives new words
from the root seed for a restored published stream. This does not prevent normal
first acquisition from deriving words for a stream that was absent from the
checkpoint. Finalization terminalizes its source session but returns
an `ACTIVE`, restartable checkpoint record. Restart creates fresh session,
registry, guard, resident arrays, and communication bindings; it never reuses
source identities or provides fallback. It rejects other versions or carrier
schemas, malformed, incomplete, partial, mixed, or mismatched communication
payloads, non-`ACTIVE` checkpoint records, and device mismatches; it does not
promise forward or backward compatibility.

Normal resident scheduler calls never checkpoint, finalize, or restart. Those
operations remain this explicit, concrete-only exact-device boundary; see the
[GPU-resident deterministic timestep](data-containers-and-gpu-foundations.md#gpu-resident-deterministic-timestep)
contract for normal-step limits.

For a lazy lifecycle-only walkthrough that does not schedule or launch physics,
see the
[GPU-resident session lifecycle source](https://github.com/Gorkowski/particula/blob/main/docs/Examples/gpu_resident_session.py).

Validate documentation with:

```bash
mkdocs build --strict
```
