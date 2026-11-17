# Simulation workflow qualification

This record qualifies the development recording/restart workflow against native
CPU, Warp CPU and RTX 5060 CUDA execution. It accompanies the
[usage guide](../../../guide/getting-started/recording-restart.md) and
[adaptive-method assessment](adaptive-methods.md). It establishes implementation
and numerical behavior for the profiles below, not experimental validity or a
GPU performance claim.

## Source and environment

The qualification extends Stage 2A `c6e8bf0967` and Stage 2B `7a53906faa` with
the forcing, reduced observation, GPU/split restart and writer implementation
in the commit introducing this record. The selected checkout was
`/tmp/particula-stage-02b`, with `PYTHONPATH=.` and its existing `.venv`.
The backend matrix includes native NumPy CPU RK23/ROS2, Warp 1.11.1 CPU and
native CUDA on RTX 5060 (sm_120; driver 13.4, toolkit 12.9).

The source package identifies itself as 0.3.1. Installed distribution metadata
in the example environment still identifies 0.2.14; no dependency upgrade was
used to hide that mismatch. Example provenance includes the executing source,
revision and dirty state. Exact restart checks package/runtime versions and
configuration, but a version string alone does not identify modified code.
Retain the source revision and study inputs alongside saved research output.

## Qualified behaviors

| Requirement | Evidence |
| --- | --- |
| Physical time, final output and resumed joins | `recording_test.py`: exact decay at requested times, final row, duplicate-free continuation and malformed-schedule rejection |
| Observation is passive for a fixed schedule | Native CPU, adaptive rejected trials, complete split cycles and prepared canonical state/RNG comparisons |
| Different output schedules retain scientific accuracy | Independent radius quadrature at `rtol=1e-5`, per-particle atol 1e-23 kg and gas atol 1e-13 kg/m³; both schedules meet radius relative error 2e-5 |
| Environmental events | `forcing_test.py`: step/linear tables, absolute stage times, right-continuous output, automatic knot landing, rejected-trial restoration and restart at a jump |
| Independent forcing accuracy | Segmented radius/reservoir equation with Sutherland viscosity; DOP853/Radau agree within 1e-10 in normalized radius; CPU RK23/ROS2 and Warp CPU/CUDA meet test-specific radius and gas targets |
| Analysis provenance and bounded storage | Units, dimensions, gas order, material densities, population measure, volume and caller-supplied configuration/numerics/source; detached arrays, opt-in slots and byte/row limits |
| Atomic writes and corruption | `analysis_io_test.py` and checkpoint tests: failed replacement preserves prior file; pending AnalysisWriter row can be retried without advancing; hashes, schemas, NPY dimensions, duplicate keys/members and unknown types reject |
| Native CPU restart | `cpu_restart_test.py`: known native processes, stochastic/condensation RNG, non-unit volumes, fields and ledgers, latent diagnostics and whole-cycle failure preservation |
| Adaptive restart | CPU RK23/ROS2/ROS3 and GPU RK23/ROS3 retain exact scaled coordinates, controller settings/counters, accepted time and table forcing; identical future schedules give bitwise continuation |
| Split restart | Nested sequential/Strang built-in condensation/dilution groups, shortened cycles, custom native fields and matching complete-cycle future state |
| Prepared restart | Dilution, Brownian coagulation, wall loss, sequence, latent condensation and an integral nucleation pulse on Warp CPU and CUDA; canonical state and RNG match after disk round trip |
| Registered prepared extensions | Custom clock/field/state and multiple managed stochastic streams survive disk restart with the caller's registry; missing registry rejects |
| Mixed CPU/GPU restart | Explicit CPUTransfer preserves host PCG64 continuation, custom fields and canonical resident state after disk restart |
| Explicit transfers and capture boundary | Prepared observer performs one setup snapshot, then declared reduced/full downloads; slot requests reject reduced mode; observation rejects active capture while qualified captured replay remains usable |
| Save functions outside capture | Existing host analysis, CPU, adaptive GPU and prepared checkpoint saves reject active native CUDA capture before replacing output; CPU archive writing does not initialize Warp |

The independent forcing tests specify all equations and reference inputs in
`forcing_test.py`; they do not derive their oracle from the production rate.
Restart tests compare future state and random continuation, not merely the
loaded starting arrays. CUDA cases ran rather than being replaced by Warp CPU.

## Formats and compatibility

Analysis uses versioned ZIP/JSON/NPY arrays. HDF5 and Zarr offer chunking but add
dependencies and more complex multi-object publication; bounded single-file
histories fit atomic replacement. Full trajectories are deliberately opt-in and
limited. Exact restart uses a distinct schema and a fixed record/enum codec,
never pickle, arbitrary imports or serialized live process callbacks.

Prepared files preserve SimulationCheckpoint v2 and ResidentCheckpoint v5
canonical bytes; lossy CPU inspection carriers are regenerated from that
authority. Native CPU and split records use schema 1. Adaptive CPU records use
schema 2 for explicit table forcing; adaptive GPU records use schema 3 for
ROS3 and exact unchanged-coordinate unscaling. Coupled GPU records use their
own schema 1 for complete event boundaries. Container schemas remain
separate from record versions. Unsupported schemas are rejected without guessed
migration. Prepared/custom adapters remain caller-supplied; loading is data
reconstruction, and resume validates the owner-specific configuration.

Opaque adaptive forcing and arbitrary split callbacks have no exact disk
identity. GPU restart requires the saved device profile, with CUDA UUID matching
for adaptive owners. Existing prepared-device constraints remain authoritative.
These restrictions are supported-profile boundaries, not claims that any Python
callback or machine can reproduce the same future trajectory.

## Executed validation

Commands ran from the selected worktree. Logs remain under ignored `.artifacts/`.

```bash
PYTHONPATH=. .venv/bin/python -m pytest particula -q -Werror \
  -m 'not slow and not performance' \
  --cov=particula --cov-report=term-missing --cov-fail-under=80
PYTHONPATH=. .venv/bin/python -m pytest scripts/tests examples_tests \
  -q -Werror --no-cov
.venv/bin/ruff check particula/ examples_tests/ scripts/ --no-fix
.venv/bin/ruff format particula/ examples_tests/ scripts/ --check
.venv/bin/python -m mypy particula/ --ignore-missing-imports
PYTHONPATH=. .venv/bin/python scripts/docs_examples.py refresh first-cpu-workflow
PYTHONPATH=. .venv/bin/python scripts/docs_topics.py
npm --prefix website run build
PYTHONPATH=. .venv/bin/python scripts/check_docs_structure.py --site-dir website/dist
```

- Full package: **10679 passed, 19 skipped, 57 deselected, 1 xfailed**;
  **91.18% coverage**, exceeding the normal 80% gate. Skipped/slow/performance
  cases are outside that execution claim. This run preceded the small archive
  capture guard; the affected save/restart suites then passed **57 tests**.
- Tooling and examples: **565 passed, 1 skipped**. The skip remains outside the
  execution claim. Standard Ruff/format passed (753 files); standard Mypy passed
  (673 sources). Untyped function bodies are outside standard Mypy's scope.
- Final combined numerical/workflow suite: **158 passed** with warnings as errors,
  including actual CUDA. All fourteen runtime modules changed by Stage 2B have **90% aggregate
  focused coverage**, using directory-based coverage discovery and explicit
  report paths. This avoids importing/unloading NumPy during dotted discovery.
- All six new workflow guide Python blocks executed, including CUDA adaptive
  restart and prepared observation/restart. The full first CPU example refreshed
  successfully; its two-case 300-DPI figure was visually inspected. The saved
  bundle includes compact plot summaries, not full restart files.
- Website: 554 pages built; 684 HTML pages, 319 API modules and 130 redirects
  verified with zero errors. Rendered structure check also passed. The final
  prepared codec matrix, including mixed host RNG, passed **23 tests**.

## Scientific and operational limits

The first example's 10 and 20 µg/m³ vapor cases reach about 136.69 and 162.13 nm
from 100 nm over 600 s, with species mass residuals below 7e-16 relative. These
are illustrative model predictions, not measurements. Conserved mass and exact
restart do not establish physical accuracy. Local adaptive tolerances are not
global error guarantees, and changing output boundaries can change a numerical
or stochastic trajectory. Compression/download cost is explicit and no workload
speedup is inferred from correctness checks.

Review examined state detachment, native measures and species maps, accepted
versus trial publication, exact coordinate restoration, forcing one-sidedness,
RNG and field continuation, atomic replacement and non-executable loading. The
capture-save gap was corrected and covered by an actual CUDA regression.
Publication and merge status belong to the plan delivery records; these results
do not assert a released package or a deployed documentation site.

## Coupled GPU event boundaries

The follow-through implementation adds `GPUCoupledSimulation`: fixed native
source/collision/removal/dilution events and a separately controlled GPU
condensation flow. It preserves persistent event buffers/RNG, explicit
sequential order and full host transfer boundaries. The controller resets
at every event boundary. Its distinct schema-1 checkpoint contains canonical
event continuation, condensation identity, scalar controls, order and work
counters; the outer archive codec remains explicit data, without pickle.

The focused coupled/numerics/archive/restart selection passed **65 tests with
one deliberate Warp-CPU capture skip** on Warp CPU and the RTX 5060. It includes
exact disk continuation with source and Brownian collisions, captured events,
rejected condensation trials without repeated event draws, unchanged caller
state and event-buffer identity, independent dilution/inventory checks,
empty-population continuation, shorter-bound rejection and faulted-cycle
observation rejection. The latter retains the last complete public clock while
rejecting further snapshots and checkpoints after partial physical work.
Changed condensation physics rejects before device access. Unsupported event
families reject before allocation. All-package Mypy and Ruff checks passed.

These tests do not substitute for the full published workloads. The cyclic
chamber still requires all 32 boxes, 8,192 slots, 600 one-second cycles, 128-event
pulses and complete second-half restart. The forcing example retains 16 boxes,
4,096 slots, 6,000 one-microsecond intervals and the full doubled-step refinement
with unchanged control times. Their updated sources and compact recording are
implemented. Both full publications were launched in a detached checkout pinned
to `480ecf2c1a`, separately from subsequent edits. The forcing publication and
figure inspection subsequently completed, as recorded below. The cyclic chamber
still requires full-workload completion and figure review. No GPU speedup is
inferred from this explicit transfer path.

The broader integration gate completed with **10,809 package tests passed**, 21
skips and one expected failure in 436.94 s; unfiltered coverage was **91.33%**.
This includes actual CUDA. The additional intentional skip is the coupled
native-capture case under the Warp-CPU fixture. Example/tooling checks passed
**554 tests with one skip** in 141.64 s. The 563-page website build and rendered
structure check passed with zero errors. A subsequent explicit checkpoint-plan
type guard passed checked-body Mypy and all **17 coupled tests**, with the same
single Warp-CPU capture skip. The full suite began before that small guard was
added; its coverage is complemented by this focused rerun, not represented as
an untouched-source final-head run. Logs are retained in
`.artifacts/gpu-coupled-package-coverage.log`,
`.artifacts/gpu-coupled-tooling-examples.log`,
`.artifacts/gpu-coupled-final-focused.log` and
`.artifacts/gpu-coupled-website.log`.

Direct `checkpoint_coupled(owner)` calls now hold the same reentrant cycle lock
as `owner.checkpoint()`. Review found that the standalone helper could otherwise
copy the event checkpoint and aggregate counters at different cycle boundaries.
A deterministic threaded regression fails against the previous unlocked helper
and passes with the lock. The full coupled test module then passed **18 tests
with one intentional Warp-CPU capture skip** in 62.25 s. Ruff, formatting and
checked-body Mypy passed for the affected modules. The reproducer and focused
logs are `.artifacts/gpu-coupled-lock-regression-before.log` and
`.artifacts/gpu-coupled-checkpoint-lock.log`. The ongoing full publications use
the already-locked owner method; this correction does not change their physics.

The subsequent unfiltered package gate ran against executable revision
`2b4bddefd8`, including the checkpoint lock and plan-type guards:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  --cov=particula --cov-report=term-missing --cov-fail-under=80
```

It passed **10,810 tests**, with 21 skips and one expected failure, in 514.17 s;
coverage was **91.34%**. Actual CUDA cases ran. Full Ruff/format checks passed
for 764 files, and ordinary Mypy passed for 681 sources. Documentation structure
reported zero errors, and the topic graph validated 33 topics, 170 resources and
35 supporting pages. Logs are `.artifacts/stage-02-current-package-coverage.log`,
`.artifacts/stage-02-current-ruff.log`, `.artifacts/stage-02-current-mypy.log`,
`.artifacts/stage-02-current-structure.log` and
`.artifacts/stage-02-current-topics.log`.

The separate respiratory-refinement diagnostics passed all three tests with
warnings as errors. They verify complete source setup reuse without time
advancement, detection of a minor-species budget defect and normalization that
does not hide a small solute behind water. These are checks of the assessment,
not completion of its full 20 s numerical comparison. The pinned full comparison
and the outstanding full example publications remain required.

## Full CUDA forcing publication

The complete `gpu-prepared-forcing` publication subsequently passed on the
RTX 5060 from pinned revision `480ecf2c1aa95da42985098c53e4ecdc66133d7b`.
Its package source hash is
`bbe6cb05017e86b19f7dc166c18cac96c0d762b5300c03990d91cd1a4e08ce8b`.
The guarded run took **3,139.79 s**, retaining all sixteen boxes, 4,096 slots
per box, 49,152 initially active particles, four materials and five gas lanes.
The coarse run used 6,000 one-microsecond cycles; the fine run used 12,000
half-microsecond cycles and `rtol=1e-7` rather than `1e-5`. Both retained the
same physical duration and 120 forcing control times.

- Maximum final per-particle total-mass relative difference: **2.32234e-7**
  against the predeclared 1% threshold.
- Maximum final water-vapor relative difference: **8.85905e-9** against 0.1%.
- Maximum per-box/species inventory residual: **1.15463e-14** coarse and
  **2.93099e-14** fine, with external vapor changes and dilution accounted for.
- Coarse adaptive work: 9,682 accepted and 1,711 rejected trials; fine work:
  17,281 accepted and 2,797 rejected trials. Both had zero domain rejections.
- Coarse coupling used 12,000 full downloads and 12,000 full uploads; the fine
  run used 24,000 of each. These measured costs preclude claiming fully resident
  execution or inferring a GPU speedup from this correctness comparison.

All saved-file hashes and scientific inputs were verified. The retained compact
report contains thirteen frames and numerical refinement/work summaries, without
full particle states. Figures were inspected before and after CPU-only replot;
the replot preserves the original CUDA execution revision, device and timestamp.
Its invocation now explicitly shows the figure, avoiding duplicate notebook
display of the helper's returned figure. The extracted download's source and
helpers matched byte-for-byte, imported from its own directory, and rendered the
saved data with GPU calls disabled and no data mutation. That download check
tested import/rendering, not a second execution of the full scientific workload.

The sparse displayed history records the supplied initial state before the first
control update, then complete cycles before the next control update. It does not
resolve all control changes or pulse peaks. The final-state refinement test is
numerical consistency evidence for the declared case, not an independent
experimental reference or an error bound at every intermediate time. The full
cyclic chamber and the remaining CPU scientific studies are still required.
The imported bundle passed the 563-page website build and rendered verification
with zero errors; see `.artifacts/coupled-forcing-website.log`. The final ZIP's
executable source and all helper bytes match the independently tested download.


## CUDA-resident material solvers and respiratory refinement

Revision `9c548d9d617a93cd23a45773426f751e9e6ed9ec` qualifies the expanded
multi-material RK23 and ROS3 GPU paths, device-owned adaptive decisions,
selective gas forcing, and deterministic parallel reservoir reductions. The
full package gate passed **10,874 tests**, with 21 skips and one expected
failure, in **331.88 s**; coverage was **91.02%**. Actual CUDA ran on the
RTX 5060. Full Ruff/format checks and Mypy (696 sources) passed. Focused
independent reservoir and dense-solve oracles passed 38 tests, including
12-material and larger-than-one-reduction-lane-group cases.

The complete 1,000-particle, twelve-material respiratory example ran to 20 s
on that revision in **286.82 s**. Its separate coarse/fine assessment completed
all **201 observations** in **1,229.39 s**, retaining the original inventory
assertions. The fine run used 100-fold tighter relative and absolute tolerances,
half the maximum step and half the observation interval. Both prospective
0.1% consistency targets passed:

- Maximum per-particle relative radius difference: **3.92121347e-6**.
- Maximum per-species phase-inventory difference: **3.22563458e-8** of that
  species' initial total inventory.
- Coarse: **8,921 accepted**, one rejected step; fine: **41,841 accepted**,
  one rejected step. Neither run had domain rejections.

The complete repository report at
`docs/data/numerics/resident-respiratory-refinement.json` retains every
observation, material identity, controls, source hashes and work counts. These
elapsed times include concurrent GPU work and are not isolated performance
benchmarks. This is same-method numerical consistency evidence;
it does not independently validate the chemistry or correct the example's
previously documented chemical-name matches. The refreshed publication records
CUDA execution, fixed revision and source hashes. The cyclic chamber has
completed its 600 s trajectory and exact checkpoint continuation; the first
publication failed during plot-data serialization, as recorded below.


### Full resident biomass publication

The 10,000-particle, three-material biomass workload completed on the same
revision and RTX 5060 in **4,148.09 s** with concurrent GPU work. It retained
0.01 s initial adjustment, 600 s preparation and all four independent 3,600 s
branches, each with 10,000 outer intervals of 0.36 s. The user-authorized model
change replaces AO2008/DNS with **ST1956 turbulent shear**, retaining
`epsilon = 0.1 m²/s³` and fluid density `1.225 kg/m³`. This is a changed collision
model, not a CUDA implementation or validation of AO2008.

All four branches passed the unchanged per-species inventory assertion
(`rtol=1e-12`, `atol=1e-30`). Each coupled branch asserted zero physical-state
uploads and downloads between process stages. Adaptive decisions remained in
native CUDA conditional graphs. Physical-state downloads were limited to
declared observations; control metadata was read at completed outer intervals.

| Coupled branch | Accepted condensation steps | Rejected steps | Domain rejections |
| --- | ---: | ---: | ---: |
| Brownian | 31,450 | 7,221 | 0 |
| Brownian + SP2016 settling | 21,902 | 3,612 | 0 |
| ST1956 turbulent shear | 27,602 | 6,101 | 0 |

The refreshed biomass bundle retains the executed source, package hash,
revision, device identity, scientific stdout and all four figures. The final
size-distribution and mixing-state figures explicitly identify turbulent shear.
These are completion and conservation checks for this illustration, not a
collision-model convergence study or experimental validation.


### Publication runner timeout correction

The full cyclic trajectory exceeded the registry's nominal two-hour limit.
Inspection found that the synchronous Jupyter client blocked the notebook
event loop while waiting for output, preventing the timeout from firing.
The publisher now uses an asynchronous kernel manager and retains explicit
kernel cleanup. A real-kernel regression (a seven-second cell with a
five-second deadline) failed before this fix and passes after it, including
verification that the timed-out kernel is released. All 23 publisher tests
pass with warnings treated as errors; website tooling passes 61 Python and
14 JavaScript tests. This runner change does not alter simulation physics.
The already-running chamber retained its original loaded runner.


### Full resident cyclic trajectory and output recovery

The first complete resident cyclic run at `9c548d9d61` passed all 600 one-second
cycles on the RTX 5060, with 32 boxes, 8,192 slots per box, 196,608 initially
active particles and twelve prescribed source pulses. It finished with 37,064
active particles. The maximum observed box/species inventory relative residual
was **1.19682042e-13**, below the unchanged `rtol=2e-12` gate. Charge, unit
counts and finite-state checks passed at all thirty observation times.

The 300-second checkpoint reproduced the complete second half with byte-exact
final state and random continuation. The original trajectory used **620,558
accepted** and **378 rejected** condensation steps, with zero domain rejections,
1,200 control readbacks and **zero physical-state coupling uploads/downloads**.
These results are retained in the captured completion stdout; the first attempt
did not publish a result bundle. Its subsequent plot-data writer rejected NumPy
arrays, and kernel cleanup released the in-memory observations. No missing
trajectory data have been reconstructed or substituted.

The observation boundary now converts detached arrays to plain lists. This
changes output representation only. A regression using the actual observation,
JSON writer and CPU-only plotting path reproduces the failure before the fix
and passes after it. Together with the publisher tests, **24 tests passed with
warnings treated as errors**. The timeout is now 14,400 seconds to accommodate
the measured full workload with enforced deadlines. The fresh full publication
from `3d3753d2b` completed in **6,307.33 s**, using identical library bytes and
scientific settings. It reproduced all work counts and accounting results above,
including exact same-device restart and zero stage transfers. The retained
**659,430-byte** data bundle contains all **31 frames** from 0 through 600 s.
The four-panel figure was inspected before and after CPU-only replot. Replot
preserved the data hash and original CUDA execution provenance; no simulation
was rerun for rendering. The initial displayed point precedes the first forcing
update, and the 20-second observations do not resolve every jump or peak. This
completion time is not an isolated performance benchmark.


The corrected runner also validated all **59 CPU-policy registry examples**
without replacing their saved publications. This check called the existing
`validate_example` path for each CPU-policy entry; all-library scientific
bytes remained fixed during that pass and the chamber rerun.

Key validation commands for the resident changes and output fixes were:

```bash
PYTHONPATH=. .venv/bin/python -m pytest particula \
  --cov=particula --cov-report=term-missing --cov-fail-under=80 -Werror -q
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python -m pytest \
  examples_tests/gpu_cyclic_chamber_example_test.py \
  scripts/tests/docs_examples_test.py -q --no-cov -Werror
.venv/bin/ruff check particula/ examples_tests/ scripts/ --no-fix
.venv/bin/ruff format particula/ examples_tests/ scripts/ --check
```


Final local documentation verification found all **89 registered bundles current**
and zero structure errors. The website built **571 pages**; rendered verification
checked 701 HTML pages, 332 API modules, 1,860 legacy anchors and 130 redirects
with zero errors. The local Stage 2B branch retains the qualified outputs;
GitHub publication remains pending direct push authorization.
