# Stage 2B execution target

Status: in progress; no completion claim.
Worktree: `/tmp/particula-stage-02b`.
Branch: `codex/stage-02b-simulation-workflow`.
Verified upstream base: `356fc054cd9a4f4ececc3c64c7144e889a789494`.
Current integration base: Stage 2A commit
`c6e8bf0967` (native CUDA adaptive condensation and measured method guide).
PR target: `uncscode/particula:main`, with the stacked Stage 2A dependency
identified explicitly. No matching PR existed at start; no PR created yet.
Merge is outside execution scope.

## Completion boundary and evidence ledger

All six work packages and acceptance criteria in
[Stage 2B](stage-02b-simulation-workflow.md) remain required. Shared physics is
Stage 1 monodisperse closed-box growth, with dilution for coupled checks.

- [ ] Duration, forcing, requested output times, variables and units contract.
- [ ] Composable detached recorder/save actions at accepted complete steps.
- [ ] Exact output landing, final outputs, coincident forcing and resumed joins.
- [x] Analysis format comparison and portable non-executable schema.
- [x] Reloadable diagnostics/optional bounded snapshots with units, dimensions,
      maps, population measure, volume, configuration and numerical provenance.
- [ ] Versioned exact restart retaining state, physics, clock, RNG, controller
      and auxiliary state for all declared CPU/GPU combinations.
- [ ] Atomic replacement/corruption rejection and complete restart matrix.
- [ ] Observer invariance and alternate-output accuracy; rejected trials,
      complete split steps, final time, resumed schedules and forcing coverage.
- [ ] Explicit GPU downloads and serialization outside CUDA capture.
- [x] Realistic first CPU example: full time series, plotted saved/reloaded data,
      exact restart and controlled scientific input variation; visually checked.
- [ ] Supported workflow documentation, broad validation, review and PR.

Unchecked items have partial implementations below; no omitted requirement is
implicitly waived. Interfaces remain concrete imports, not package-root exports.

## Implemented contracts and current evidence

`recording.Recorder` retains selected detached observations with units, axes,
full gas names/molar masses/partitioning, density, volume and PDF metadata.
Full slots are opt-in and storage is bounded. `run_until` lands on requested
output/final times and skips older requests on resume without duplicate rows.
Identical schedules preserve trajectory and RNG. Adaptive rejections emit no
rows; complete Strang cycles preserve the independently expected growth/loss
budget. Different output schedules meet an independent radius-quadrature target.
Forcing/event integration and prepared-owner observation are still pending.

Analysis uses versioned ZIP/JSON/NPY with payload hashes, byte limits, unique
members/keys and NPY header checks before allocation. HDF5/Zarr offer chunked
access but add dependencies/atomic multi-object complexity; the chosen format
is for bounded single-file histories. Atomic temporary-file replacement protects
the last complete file. Loaders do not use pickle or import file-named types.
Analysis and restart are separate schemas and cannot substitute for one another.

Prepared disk restart transports SimulationCheckpoint v2 and ResidentCheckpoint
v5 using fixed allowed records. Canonical immutable bytes remain authority;
CPU inspection carriers are regenerated before saving. Exact package/NumPy/Warp
and device/process compatibility remain required. Actual Warp CPU and RTX 5060
checks preserve future canonical state/RNG after disk round trips for dilution,
Brownian coagulation, wall loss, sequence, latent condensation and an integral
nucleation pulse. Custom extension record qualification remains incomplete.

Native `CPUSimulation` stages state, process diagnostics and CPU RNG together;
it never launches Warp. Known native configuration signatures retain options
that the GPU planner cannot represent. CPU checkpoints include native carriers,
extension fields and additive ledgers, RNG, clock/cycle count and latent-energy
diagnostics. Tests cover native stochastic processes, randomized staggered
condensation, non-unit volumes, detached data, configuration rejection and
later-component failure rollback. Unknown custom physics is rejected.

Adaptive condensation checkpoints retain scaled coordinates, physical inputs,
tolerances, accepted clock, next step, last error, work counters and viscosity.
RK23 and ROS2 resume bitwise on identical future schedules. Arbitrary callable
forcing has no supported disk identity. Split-owner and declarative forcing
restart remain pending, not covered by the adaptive checkpoint claim.

## First simulation

`docs/examples/fundamentals/first_cpu_workflow.py` replaces the exaggerated
100 kg/m3 guide case with 600 s organic growth at 10 and 20 micrograms/m3 vapor,
100 nm initial radius and 1000 particles/cm3. Each full workload records 61
rows, saves/reloads analysis, restarts at 300 s and checks exact future state.
Final radii are 136.6858 and 162.1279 nm; relative budget residuals are below
7e-16. These are illustrative model predictions, not experimental validation.
Both guide reload/restart snippets execute against actual saved output.

The registered `first-cpu-workflow` bundle contains compact plotted summaries
and one visually inspected 300-DPI figure. Restart files stay in temporary
publication storage and are not committed. The report records executing source
version 0.3.1; the environment's installed metadata still says 0.2.14 in the
standard manifest. Source hash/revision/dirty state distinguish the actual
checkout. Dependencies were not upgraded to conceal that metadata mismatch.

## Current validation and remaining work

Combined numerical/workflow/tutorial suite: **81 passed with -Werror**, including
actual CUDA, using the selected worktree's `.venv` with `PYTHONPATH=.`. Coverage
of the seven new runtime modules: **87% aggregate**, normal 80% gate passed;
per-module coverage is analysis 74%, archive 86%, prepared codec 91%, native
configuration 90%, CPU restart 86%, CPU owner 88%, recorder 90%.

Coverage was collected with `coverage run --source=particula/execution` and
reported with the seven explicit source paths. Dotted module discovery caused
coverage to import/unload NumPy before collection; directory discovery avoids
that failure without suppressing warnings or changing the coverage gate.
Ruff/format and checked-body Mypy passed for new runtime sources. The complete
example refresh succeeded with local Jupyter socket access. Astro used an
isolated dependency copy, built 544 pages, and verified API/redirect targets
without errors. Full-package coverage and final review remain pending.

Next: forcing/event semantics and checkpoint identity; prepared GPU observations
with explicit transfer costs and capture limits; split restart; GPU extension
codecs; complete workflow/method guides; broad repository validation and PRs.
Stage 2A native CUDA adaptive qualification and the measured method guide are
now integrated. Its broad package/tooling checks passed; shared workflow
compatibility and both PRs remain open. Neither stage is complete.

After rebasing onto that numerical implementation, the recording/analysis/disk
restart/native CPU restart/native GPU subset passed **70 tests with -Werror**,
including actual CUDA. The two first-workflow example tests also passed. Both
documentation topic additions were preserved and
the combined graph validates. The Stage 2B workflow commit is `2cd525c9de`;
no duplicate branch or PR was created.
