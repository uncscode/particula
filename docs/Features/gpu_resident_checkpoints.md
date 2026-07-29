# GPU resident checkpoints

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

Restart compatibility is intentionally exact and fail-closed. The implementation
accepts only `ResidentCheckpoint` records with schema version `1`, carrier type
`"ResidentSession"`, lifecycle `ACTIVE`, complete valid canonical payload
descriptors, and an exactly equal target `Device`. Finalization terminalizes its
source session but returns an `ACTIVE`, restartable checkpoint record. Restart
rejects other versions or carrier schemas, malformed or incomplete payloads,
non-`ACTIVE` checkpoint records, and device mismatches; it does not promise
forward or backward compatibility.

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
