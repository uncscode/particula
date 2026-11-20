# Simulation workflow qualification

This record qualifies the development recording/restart workflow against native
CPU, Warp CPU and RTX 5060 CUDA execution. It accompanies the
[usage guide](../../../guide/getting-started/recording-restart.md) and
[adaptive-method assessment](../condensation/adaptive/index.md). It establishes implementation
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

The [coupled GPU event record](coupled-gpu-events.md) gives the checkpoint comparisons, reproduction commands and evidence limits.

## Full CUDA forcing publication

[Full CUDA forcing publication in the study](../../examples/coupled-populations/forcing/forcing-refinement.md#full-cuda-forcing-publication)



## CUDA-resident material solvers and respiratory refinement

Revision `9c548d9d617a93cd23a45773426f751e9e6ed9ec` qualifies the expanded
multi-material RK23 and ROS3 GPU paths, device-owned adaptive decisions,
selective gas forcing, and deterministic parallel reservoir reductions. The
full package gate passed **10,874 tests**, with 21 skips and one expected
failure, in **331.88 s**; coverage was **91.02%**. Actual CUDA ran on the
RTX 5060. Full Ruff/format checks and Mypy (696 sources) passed. Focused
independent reservoir and dense-solve oracles passed 38 tests, including
12-material and larger-than-one-reduction-lane-group cases.

[Read this evidence in the study.](../../examples/coupled-populations/emissions/respiratory-refinement.md#cuda-resident-material-solvers-and-respiratory-refinement)



### Full resident biomass publication

<a id="biomass-rerun-with-adaptive-gpu-coagulation"></a>
[Full resident biomass publication in the study](../../examples/coupled-populations/emissions/biomass-evidence.md#full-resident-biomass-publication)


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

[Full resident cyclic trajectory and output recovery in the study](../../examples/coupled-populations/chambers/cyclic-chamber-evidence.md#full-resident-cyclic-trajectory-and-output-recovery)



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
