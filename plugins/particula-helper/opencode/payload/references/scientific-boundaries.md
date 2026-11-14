# Scientific and execution boundaries

These are contribution guardrails for the native 0.3.0 data model. Resolve all
repository paths against the user's checkout and verify its current owning
source before changing a contract. This summary is not an API signature manual.

## Native state

Read `docs/guide/data-model/index.md`, `docs/guide/data-model/ownership.md`, and
`docs/guide/migration/v0.3.0/index.md` for schemas and migration details.

- Construct `Aerosol` from native `ParticleData`, full ordered `GasData`, and
  `EnvironmentData`. Legacy particle/species/atmosphere facades are removed.
- Resolved concentration is represented count, discrete concentration is number
  density, and continuous PDF is dN/dr. Account for representative volume in
  density and extensive-inventory comparisons. GPU PDF remains unsupported.
- Particle-material width and full-gas width are independent. Preserve ordered
  names, inert lanes, partitioning masks and explicit process species maps.
  Physical configuration belongs to processes.
- `Aerosol` holds containers by identity. Validated individual/coordinated
  replacement is distinct from raw array mutation. Layout changes require
  rebuilding process maps and prepared GPU bindings; there is no auto-rebinding.
- The five native dynamics runnables support batched CPU execution through the
  shared transactional box loop. Direct strategy methods remain single-box.
- Fixed slots are active only with finite positive concentration, finite
  nonnegative masses with positive total mass, and finite charge. Free slots
  have exactly zero concentration, all masses and charge. Reject other states.
  Preserve fixed capacity; never silently resize, compact or truncate demand.

## Process behavior

Read the owning guide under `docs/guide/processes/` and its tests.
Preserve each process's supported distributions and scientific assumptions.
Binned coagulation retains its fixed-composition interpretation; it is not a
general radius-dependent multispecies composition model.

Direct GPU condensation uses exactly four equal substeps, device vapor-pressure
refresh, inventory-limited transfers and sequential gas coupling. That bounded
contract does not establish CPU-strategy parity or adaptive stepping. Charged
wall-loss image enhancement remains active at zero wall potential. Slot
exhaustion resolves viable resampling before enabled volume-scaling fallback.
Inspect the owning contract before changing their mutation or no-op behavior.

Validate preflight rejection and failure atomicity at their documented boundary.
Successful earlier substeps may remain when a later substep fails. Do not claim
whole-call rollback merely because preflight is read-only.

## GPU state and lifecycle

Read `docs/guide/gpu/prepared.md`, `support.md`, `transfers.md`,
`slot-exhaustion.md`, and the
`resident/checkpoints.md` and `resident/graph-capture.md` guides in that folder.

- Public runnable `prepare()` owns explicit upload and fixed resident resources;
  `snapshot()` downloads detached native CPU state. Supported ordered subsets
  contain at most one occurrence of each of five process families. Check exact
  configuration support before recommending this path. Condensation remains
  four fixed substeps; Brownian requires equal active resolved counts, including
  after any source or sink. No hidden backend fallback is provided.
- Prepared forcing is explicit, diagnostics reduce without full particle
  download, and checkpoints preserve exact continuation rather than just
  analysis state. Capture records without advancing and requires native CUDA.
- Direct step entry points import from `particula.gpu.kernels`; conversion helpers
  from `particula.gpu`. Keep concrete-only configuration/sidecar/lifecycle names
  out of package exports unless an API change is explicitly in scope.
- Callers own devices, stable-shape buffers, transfers and synchronization.
  Preserve explicit boundaries; do not add hidden transfers or CPU fallback.
  A separately documented opt-in fallback policy does not imply fallback in
  direct kernels, resident execution, checkpointing or graph capture.
- Gas names are CPU-owned metadata. Preserve ordered names externally for a
  semantic direct restore; prepared owners retain these names automatically.
  GPU vapor pressure is derived state, omitted from CPU
  inspection/restore carriers.
- Supplied coagulation/wall-loss RNG sidecars persist across calls. Repeating
  a root seed does not reset them; initialization/reset is deliberate.
- Resident resources require exact session/registry/closed-guard bindings.
  Normal steps do not reseed, checkpoint, restart or recreate bindings.
- Schema-v5 checkpoints own immutable canonical device bytes and RNG
  continuation metadata. CPU inspection carriers are lossy. Restart is explicit
  with fresh identities on an exactly compatible device, with no inferred old
  schema migration or cross-device continuation.
- Preserve read-only rejection guarantees. After an asynchronous writer launches,
  rollback is not promised; honor documented fault and terminal transitions.
- Native graph capture requires CUDA. Preserve qualification, authenticated
  handle provenance, replay admission, invalidation, retirement and teardown.
  There is no Warp-CPU capture emulation, automatic recapture or retry.

## Evidence

Use the maintained records in `docs/contribute/validation/`, especially
`gpu/prepared-rtx5060.md`, `gpu/graph-capture-qualification.md`,
`gpu/resident-benchmarks.md`,
`gpu/graph-capture-profiling.md`, `numerics/mass-precision.md`, and
`releases/v0.3.0.md`. Implementation validation, release approval and performance
qualification are separate. Missing measurements remain unavailable.

The controlled coagulation output is
`.artifacts/benchmarks/gpu_benchmark_results.json`, an ignored generated output
that is not retained in the repository. Optional runs may recreate the directory.
The viewer's declared snapshot is `docs/data/gpu_benchmark_results.json`. Neither is resident-capture evidence.
Resident measurements use `.artifacts/benchmarks/resident_capture_comparison.json`
and the designated-device/final-revision qualification gates. Historical
measurements do not become current by changing package metadata. Profiler
intervals and host-launch timing are not interchangeable throughput measures.

The prepared campaign is `.artifacts/gpu-campaign/`; its compact publication
data is `docs/data/gpu/rtx5060.json`. Publish only a complete qualified matrix,
matching executable digest, exact common GPU continuation and separate full-array
CPU/CUDA equivalence for CPU timing comparisons. Long CUDA illustrations are
manually refreshed; CI does not execute them. Do not relabel Warp CPU as GPU.
