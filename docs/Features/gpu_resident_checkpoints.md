# GPU resident checkpoints

The direct-import-only checkpoint boundary is available from
`particula.execution.checkpoint`; it is deliberately not exported by
`particula.execution` or the top-level package. Create a
`ResidentCheckpointController` for one active `ResidentSession`, its pinned
`GPUResourceRegistry`, and its `ResidentStepGuard`, or use the session's
`checkpoint(registry, guard)` and `finalize(registry, guard)` methods.

`checkpoint()` is nonterminal and returns a fresh immutable host snapshot.
`finalize()` is terminal and idempotent: after its first successful call the
session is `FINALIZED` and later calls return the cached snapshot. Checkpoints
are explicit in-memory, same-device recovery only. They do not serialize to
disk, select or migrate a device, synchronize implicitly during restart, or
provide rollback after a device writer has launched.

The snapshot owns immutable canonical bytes for primary arrays and acquired
sidecars, plus detached CPU inspection carriers. Inspection `GasData` is
intentionally lossy because CPU gas carriers do not contain GPU vapor pressure;
restart uses canonical bytes and restores vapor pressure exactly. Snapshotting
requires approximately one additional host copy of resident payload bytes plus
the detached inspection copies. Restart explicitly requires the compatible
target `Device` through `restart_resident_session(checkpoint, device)`.

Validate documentation with:

```bash
mkdocs build --strict
```
