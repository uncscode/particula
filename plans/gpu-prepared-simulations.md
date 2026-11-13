# GPU prepared simulations and illustrated CUDA studies

Accepted implementation plan, 2026-11-13. Branch:
`codex/gpu-prepared-simulations`; one final PR to `uncscode/particula:main`.

## Outcome and scope

Connect native `Aerosol` and existing runnables to a prepared GPU session:
upload once, execute resident physics, explicitly update forcing between run
segments, and download detached CPU snapshots for analysis. Support ordered
subsets of condensation, Brownian coagulation, dilution, wall loss and
nucleation, with one occurrence per family and explicit capability rejection.
Keep existing CPU execution and direct GPU APIs working. No hidden fallback,
new PDF algorithm, unequal-multiplicity coagulation, or multi-GPU support.

Public preparation retains process configuration, timestep, logical box IDs,
and random-stream identities. Session operations: `step`, `run`,
`synchronize`, `snapshot`, `update_environment`, `update_gas`, reduced
diagnostics, `checkpoint`, `capture`, and context-managed cleanup. Sequence
`resume` checks configuration and exact-device compatibility. A snapshot is
detached CPU analysis state; a checkpoint preserves canonical continuation
bytes. Preserve process order and insert necessary thermodynamic refreshes.
Frozen configuration does not follow later edits to CPU process objects.

Preserve float64, species mappings including independent material/gas widths,
ordered names, inert lanes, represented counts and physical volumes. Reject
unsupported physics before upload. Guard evolving equal-count compatibility
before coagulation; do not silently reinterpret nucleation/dilution output.
Device failure status stops dependent work, surfaces at observation, and does
not promise rollback. No state readbacks or allocations inside prepared steps.

## Phase commits and completion ledger

- [x] 1. `test(gpu): make CUDA qualification explicit`: normalize aliases,
  correctly mark parameter cases, require nonempty CUDA execution and reject
  unexpected skips. Record hardware/runtime provenance.
- [x] 2. `feat(execution): add prepared simulation ownership`: upload,
  detached complete snapshots, synchronization and lifecycle/cleanup.
- [x] 3. `feat(execution): prepare supported runnable processes`: faithful
  configuration translation, persistent resources, nondefault coefficients,
  explicit capabilities and independent per-process validation.
- [x] 4. `feat(execution): compose resident process sequences`: ordered
  subsets, thermodynamic freshness, forcing, diagnostics, runtime guards and
  RNG continuity; preserve the existing complete canonical scheduler.
- [x] 5. `perf(gpu): parallelize Brownian majorant evaluation`: parallel
  properties and bounded tiled pair maxima, ascending active ranks and original
  sequential selection/RNG semantics. No full N-by-N allocation; other
  mechanisms unchanged. Retain quadratic complexity honestly in documentation.
- [ ] 6. `feat(execution): integrate prepared restart and CUDA capture`:
  checkpoint/resume, explicit capture with no duplicate timestep, provenance,
  invalidation, faulting and teardown tests.
- [ ] 7. `test(gpu): add reproducible scaling and variability studies`:
  controlled workloads, scientific assertions, process-isolated budgets,
  reproducible measurements and provenance.
- [ ] 8. `docs(gpu): publish illustrated CUDA tutorials`: convert all eight
  existing GPU examples to percent-cell Python, add substantive studies,
  refresh complete CUDA workloads and visually review figures/rendered pages.
- [ ] 9. `docs(gpu): record RTX qualification and feature guidance`: final
  evidence, capability/migration guides, accurate GPU labeling, bounded Recent
  news entry, complete validation and a single PR.

Use follow-up fix commits when necessary; each phase has its own validation.
Do not infer completion from this checklist: verify executable behavior,
current results, plots and PR state before closing the goal.

## Required study matrix

Principal scenarios: two particle materials, three gases including an inert
lane. Include a four-material/five-gas variation and 25/75/100% occupancy.
Report active count separately from capacity; nucleation needs free slots.

| Study | Workload | Required output |
| --- | --- | --- |
| Particle scaling | B=1; N=1024,16384,131072,1048576; condensation/dilution | Size distributions, conservation, throughput and memory |
| Brownian scaling | B=1; N=128,512,2048,8192,16384 | Collisions, evolving sizes, old/new timing, scaling limit |
| Box scaling | B=1,8,64,512,1024; N=1024 | Aggregate/per-box throughput, memory, distributions |
| Heterogeneous boxes | B=64; N=4096 | Temperature/gas/volume/composition responses, heatmaps, inventories |
| Time-varying forcing | B=16; N=4096 | Cooling/heating/gas histories, distributions and source/sink accounting |
| Stochastic variation | 16 seeds; B=16; N=1024 | Distribution means/bands, collisions and number evolution |
| Full lifecycle | Compatible multiprocess workloads including >1M aggregate particles | Snapshot isolation, restart and captured/uncaptured agreement |

Use short large-scaling trajectories and longer scientific evolution studies;
require nonzero intended physics. Three warmups and seven timing samples,
restoring identical state and RNG outside timed intervals. Measure setup/upload,
resident steps, download and end-to-end time separately. Report median, p95,
spread, actual work and memory method. Compare native CPU, direct CUDA,
prepared CUDA and captured CUDA only where scientific semantics match; Warp
CPU is separately labeled regression evidence, never GPU performance.

Limit device allocation to min(6 GiB, 75% available VRAM), include scratch and
capture/checkpoint costs, isolate each case in a subprocess with a 10-minute
limit and the final campaign with an 8-hour limit. Required incomplete cases
remain pending rather than being silently downsized. Store large raw output and
profiler traces outside Git; commit compact measurements, provenance and
publication figures. Use Nsight Systems on representative cases and attempt
Nsight Compute without changing system security settings.

## Documentation and acceptance

All eight existing GPU scripts currently have no percent cells or graphics.
Convert to `# %%` / `# %% [markdown]`, preserve importable test helpers, replace
status-only publication invocations with meaningful figures. Use
`website/examples.json` and saved `website/results/` bundles; CUDA-required
refreshes never fall back. Site builds render saved results without running
simulations. Review plots and full desktop/mobile pages. Label CUDA evidence
with GPU model; retain Warp CPU regressions and the exact `cuda` marker.

Run adjacent tests per phase, full source/example/script suites, Ruff, mypy,
required coverage, docs structure/freshness, publishing tests and Starlight
build. Final RTX qualification must execute actual required CUDA/capture
cases with no false skips. Independently check per-box/per-species conservation,
deterministic parity, stochastic expectations and failure boundaries. Benchmark
against the completed executable revision; record input hashes and rerun when
executable changes invalidate evidence. Commit result bundles with sources.

Push the completed branch to origin and create one upstream PR with API usage,
capability matrix, phase map, examples, measured RTX results, exact commands,
limits and validation. Follow hosted CI and fix failures. Do not merge/release.

## Starting evidence

Base revision: `5e930e026`; clean main. Host RTX 5060, 8151 MiB, driver
615.78.08; Python 3.12.12, Warp 1.11.1, CUDA available on `cuda:0`.
Restricted tool execution cannot see the GPU; CUDA work needs host execution.
Native full-loop/checkpoint/coagulation cases selected with `-k cuda` passed
176 tests. Their missing CUDA markers made `-m cuda` select zero cases.
Captured-loop suite: 8 passed, 4 false skips due to `cuda:0` string membership
against `["cpu", "cuda"]`. These are baseline observations, not final evidence.

## Phase evidence

Phase 1: `python scripts/qualify_gpu.py` executed **329 passed, zero skipped**
on the RTX 5060. JSON/XML provenance is under `.artifacts/qualification/`.
Correcting the alias gate exposed a stale resource reference and a stochastic
fixture with one active particle per box (zero collision opportunities).
The coagulation qualification now explicitly uses 64 partners and total number
density 1e15 m^-3; existing activity, RNG-continuation and conservation assertions
all pass. This is baseline qualification, not final feature/performance evidence.

Phase 2: owner, dedicated CUDA stream, detached snapshots, explicit observation,
faulting and context cleanup implemented in `execution/prepared.py`. This is
still internally constructed until phase 3 connects validated runnables.
Prepared/native-resident/conversion suites: **167 passed** on the host with
CUDA and Warp CPU. Ruff and targeted mypy passed. No transfer, synchronization
or allocation is allowed by the enqueue regression; all snapshot fields are
checked for isolation. Closing performs no download.

Phase 3: all five selected families now have internal immutable translation
plans and retained device resources. Condensation forwards nondefault transport
coefficients and explicitly limits activity, Kelvin density and vapor models to
representable configurations. Nucleation retains maps, scaling and resampling
bounds, rejects charged/weighted injection, and observes direct-kernel status.
Its existing integral unit-event restriction remains explicit. Public runnable
integration and persistent cross-step failure gating are phase 4 work; the
internal compiler is not yet a complete public simulation API.
Prepared/condensation/process-adapter tests: **165 passed** on the host, including
actual CUDA and separately parameterized Warp CPU. Independent checks cover
mass/count/charge preservation, active condensation, two-event nucleation,
capacity resampling, configuration freezing and pre-upload capability rejection.

Phase 4: public runnable/sequence `prepare`, ordered subsets, explicit owned
forcing uploads, derived saturation refresh, detached reduced diagnostics,
last-step latent energy, and persistent device failure admission are connected.
Guarded writers reuse the same device physics as direct APIs. Runtime unequal
counts stop Brownian selection before RNG consumption. First nucleation error
status survives later queued steps, including resampling failures. Observation
faults the session; prior successful work is not rolled back. Forcing is an
explicit synchronization boundary; ordinary stepping performs no device
allocation, transfer or status readback. Without a condensation law, saturation
remains an externally prescribed environment input.
Direct/prepared process regression set: **827 passed**. Following the final
status-retention correction, prepared/nucleation regressions: **124 passed**.
Public export and runnable regressions: **19 passed**, including fresh-process
checks that public result/owner imports do not import Warp. Targeted Ruff and
mypy passed. CUDA and Warp CPU are distinct parametrizations. Scientific checks
cover multibox forcing, raw-count versus discrete-density reductions, latent
energy, preserved gas order, no-write/no-RNG behavior after failure and stable
logical-box RNG identities across run segmentation and physical permutation.

Phase 5: Brownian-only preparation now computes per-slot transport properties
in parallel, compacts active indices in ascending order, and evaluates exact
pair maxima using at most 4096 float64 partials per box. Selection shares the
original sequential RNG/acceptance function; other mechanism masks retain their
original path. No dense pair matrix is allocated; pair work remains quadratic.
Coagulation/native/validation/parallel/prepared regressions: **1377 passed** on
CUDA and separately parameterized Warp CPU. Byte-exact legacy continuation,
independent NumPy maxima, species mass/charge conservation and a 65537-slot
memory-bound check pass. Targeted Ruff and mypy pass.
A preliminary RTX probe (three boxes, full occupancy, float64, restored primary
state and RNG, three warmups/seven samples) measured legacy/parallel median
resident steps of 10.90/0.210 ms at N=128, 158.65/1.476 ms at N=512,
2556.95/20.641 ms at N=2048, and 40653.86/178.320 ms at N=8192.
These are old-versus-new GPU execution timings, not CPU comparisons, setup costs,
or final campaign evidence. The complete required study matrix remains phase 7.
