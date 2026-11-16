# Stage 2B delivery and evidence

Status: implemented, locally validated and reviewed; final PR delivery pending.
Worktree: `/tmp/particula-stage-02b`.
Branch: `codex/stage-02b-simulation-workflow`.
Verified upstream main: `356fc054cd9a4f4ececc3c64c7144e889a789494`.
Dependency: Stage 2A [PR #1659](https://github.com/uncscode/particula/pull/1659).
PR target: `uncscode/particula:codex/stage-02a-numerical-integration`, so the
workflow diff is independently reviewable. Merge is outside execution scope.

## Completion boundary

All six work packages and acceptance criteria in
[Stage 2B](stage-02b-simulation-workflow.md) are covered by the declared profiles.
Shared physics is Stage 1 monodisperse closed-box growth, with dilution for
coupled checks. Concrete imports preserve existing public runnable APIs.

- [x] Duration, forcing, requested output times, variables and units contract.
- [x] Composable detached recorder/save actions at accepted complete steps.
- [x] Exact output landing, final outputs, coincident forcing and resumed joins.
- [x] Analysis format comparison and portable non-executable schema.
- [x] Reloadable diagnostics/optional bounded snapshots with units, dimensions,
      maps, population measure, volume, configuration and numerical provenance.
- [x] Versioned exact restart retaining state, physics, clock, RNG, controller
      and auxiliary state for declared CPU/GPU combinations.
- [x] Atomic replacement/corruption rejection and supported restart matrix.
- [x] Observer invariance and alternate-output accuracy; rejected trials,
      complete split steps, final time, resumed schedules and forcing coverage.
- [x] Explicit GPU downloads and serialization outside CUDA capture.
- [x] Realistic first CPU example: full time series, plotted saved/reloaded data,
      exact restart and controlled scientific input variation; visually checked.
- [x] Supported workflow documentation, broad validation and implementation review.
- [ ] Commit, push and separate dependent PR.

## Authoritative implementation and evidence

The [workflow guide](../docs/guide/getting-started/recording-restart.md) owns
supported usage, output timing, forcing, transfer costs and restart boundaries.
The [qualification record](../docs/contribute/validation/numerics/simulation-workflow.md)
retains the requirement-to-test matrix, format decision, scientific reference
limits, commands and results. The
[first CPU experiment](../docs/examples/fundamentals/first_cpu_workflow.py)
is the complete run/record/plot/save/load/restart/input-variation demonstration.

Implemented owners are native `CPUSimulation`, CPU RK23/ROS2 condensation,
GPU RK23 condensation, complete-cycle deterministic split simulation and existing
prepared execution. `EnvironmentForcing` tables impose one-sided step/linear
knots. `Recorder` and `AnalysisWriter` observe accepted complete work;
`PreparedObserver` explicitly chooses reduced or full downloads on the fixed
grid. Analysis and restart use separate versioned non-executable archives.

Exact continuation is tested for native CPU process/RNG/field state, adaptive
scaled coordinates/controller/forcing, nested built-in split flows, six prepared
process profiles, registered GPU state/clock/fields/multiple RNG streams, and
explicit mixed CPU/GPU host RNG. Caller-supplied compatible physics and registries
remain required. Arbitrary split callbacks and opaque forcing have no exact disk
identity; unsupported schemas, changed devices or physics reject. These are the
explicit supported-profile boundaries required by the plan, not omitted claims.

## Validation and review

- Combined package: **10679 passed, 19 skipped, 57 deselected, 1 xfailed**;
  **91.18% coverage** with normal 80% gate and warnings as errors. Actual CUDA ran.
- Tooling and examples: **565 passed, 1 skipped** with warnings as errors.
- Final combined numerical/workflow set: **158 passed**; all fourteen changed
  runtime modules have **90% aggregate coverage**. Capture rejection and mixed
  CPU/GPU disk continuation are included.
- Save/restart subset after the capture guard: **57 passed**; final prepared
  codec matrix including mixed host RNG: **23 passed**.
- Full Ruff and format checks passed (753 files), standard Mypy passed
  (673 sources). No numerical tolerance or coverage gate was weakened.
- All six workflow guide blocks executed, including real CUDA. The complete
  first CPU example refreshed; its 300-DPI figure was inspected, and saved
  summaries exclude full restart state. New navigation/topic/structure checks
  passed. Final site build: **554 pages**; **684 HTML pages, 319 API modules
  and 130 redirects** verified with zero errors; rendered structure also passed.

Review covered the full workflow diff and owning numerical/lifecycle contracts,
with particular attention to no partial trial output, native species/volume
metadata, detached histories, exact coordinate and RNG continuation, one-sided
forcing and archive safety. One finding was fixed: already-created host data
could be serialized inside active CUDA capture. The shared archive writer now
rejects that before file replacement without initializing CUDA for CPU output.
The actual-CUDA regression covers all four save APIs and preserves the old file.
No unresolved implementation finding remains in the declared profile.

Stage 2A's source-hashed method measurements remain unchanged; workflow hooks
are separately qualified by forcing and continuation tests. Hosted PR status
will be reported separately from local validation; neither local checks nor PR
creation claim a merge, release or production documentation deployment.
