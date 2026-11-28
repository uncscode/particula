# Scientific and execution boundaries

These are contribution guardrails for native 0.4.0 particle populations. Resolve all
repository paths against the user's checkout and verify its current owning
source before changing a contract. This summary is not an API signature manual.

## Native state

Read `docs/guide/data-model/index.md`, `docs/guide/data-model/ownership.md`, and
`docs/guide/migration/v0.4.0/index.md` for the 0.3.6 migration. Use the older
0.3.0 migration guide for the move from facades to native containers.

- Construct `Aerosol` from native `ParticleData`, full ordered `GasData`, and
  `EnvironmentData`. Legacy particle/species/atmosphere facades are removed.
- In 0.4.0, raw `ParticleData.population` holds counts per discrete bin or
  resolved slot, or represented dN/dr (m⁻¹) for a continuous radius PDF.
  Divide by per-box represented air volume for slot density. Integrate PDFs
  on their radius nodes in metres. Read `number_density`, `species_mass_density`
  (kg/m³ air) and `species_mass_inventory` (kg per box) for physical comparisons.
  Gas concentration is kg/m³ air. Check both a sub-unit and a larger sample.
- To import measurements, use builder `set_number_density` with units or the
  CPU instance's SI `set_slot_concentration_density(values, volume=...)`.
  For a raw writer use the owned `population` array. Derived reads detach.
  Transfer raw bin/resolved counts and their sample volumes to Warp. Native
  radius PDFs stay on CPU. Confirm each process's multiplicity requirements.
- Use `particle_population` and `volume` for new recordings. Read historical
  analysis files via `load_analysis` and inspect returned source units and
  schema. Resume exact checkpoints only under the owning version, schema,
  runtime and device contract.
- Particle-material width and full-gas width are independent. Preserve ordered
  names, inert lanes, partitioning masks and explicit process species maps.
  Physical configuration belongs to processes.
- `Aerosol` holds containers by identity. Validated individual/coordinated
  replacement is distinct from raw array mutation. Layout changes require
  rebuilding process maps and prepared GPU bindings; there is no auto-rebinding.
- The five native dynamics runnables support batched CPU execution through the
  shared transactional box loop. Direct strategy methods remain single-box.
- Fixed slots are active only with finite positive represented count, finite
  nonnegative masses with positive total mass, and finite charge. Free slots
  have exactly zero count, masses and charge. Reject other states.
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
- Schema-v6 resident checkpoints own immutable canonical device bytes and RNG
  continuation metadata. CPU inspection carriers are lossy. Restart is explicit
  with fresh identities on an exactly compatible device, with no inferred old
  schema migration or cross-device continuation.
- Preserve read-only rejection guarantees. After an asynchronous writer launches,
  rollback is not promised; honor documented fault and terminal transitions.
- Native graph capture requires CUDA. Preserve qualification, authenticated
  handle provenance, replay admission, invalidation, retirement and teardown.
  There is no Warp-CPU capture emulation, automatic recapture or retry.

## Evidence

Read the scientific comparison and its qualification record together. Under
`docs/studies/`, the prepared GPU campaign lives in `benchmarks/prepared-scaling/`,
the mass-precision decision in `validation/material-properties/mass-precision/`,
and the graph-capture gates, profiling protocol and native API release record in
`validation/workflows-and-releases/`. Each record identifies its evidence and
open gates. Missing measurements remain unavailable.

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
