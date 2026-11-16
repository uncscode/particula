# Stage 2B delivery and evidence

Status: core workflow implemented, locally validated and reviewed; delivered
in [PR #1660](https://github.com/uncscode/particula/pull/1660). The added
[cloud/droplet examples](stage-02-cloud-droplet-examples.md) are locally complete;
the later commits are published, with current hosted checks and review
readiness tracked in that PR.
Worktree: `/tmp/particula-stage-02b`.
Branch: `codex/stage-02b-simulation-workflow`.
Verified upstream main: `356fc054cd9a4f4ececc3c64c7144e889a789494`.
Dependency: Stage 2A [PR #1659](https://github.com/uncscode/particula/pull/1659).
PR target: `uncscode/particula:codex/stage-02a-numerical-integration`, so the
workflow diff is independently reviewable. Merge is outside execution scope.

## Completion boundary

The initial core delivery covered all six work packages and acceptance criteria
in [Stage 2B](stage-02b-simulation-workflow.md) for its declared profiles.
The expanded cloud/droplet implementation and full scientific runs are locally
complete and published. The linked PR records the hosted checks and readiness
transition, which requires those checks to pass.
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
- [x] Commit, push and separate dependent draft PR.
- [x] User-added cloud/droplet migrations and stability demonstrations, including
      full workloads, refinement, restart and output review.

## Authoritative implementation and evidence

The [workflow guide](../docs/guide/getting-started/recording-restart.md) owns
supported usage, output timing, forcing, transfer costs and restart boundaries.
The [qualification record](../docs/contribute/validation/numerics/simulation-workflow.md)
retains the requirement-to-test matrix, format decision, scientific reference
limits, commands and results. The
[first CPU experiment](../docs/examples/fundamentals/first_cpu_workflow.py)
is the complete run/record/plot/save/load/restart/input-variation demonstration.

Implemented owners now include native `CPUSimulation`, CPU RK23/ROS2/ROS3
condensation, GPU RK23/ROS3 condensation, complete-cycle deterministic split
simulation, prepared execution and `GPUCoupledSimulation`. Its default mode uses
explicit full-state transfers between fixed events and adaptive condensation.
With `resident=True`, particle/gas state and process handoffs remain on CUDA,
and conditional graphs own adaptive acceptance and step-size decisions. CPU
orchestration still supplies the fixed outer schedule and requested outputs;
the entire scheduled simulation is not one captured graph.
`EnvironmentForcing` tables impose one-sided step/linear
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

## Initial core validation and review

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

## Current follow-through gates

The earlier transferred-execution gate at revision `2b4bddefd8` passed 10,810 tests
with 91.34% unfiltered coverage, including actual CUDA. The
[workflow qualification record](../docs/contribute/validation/numerics/simulation-workflow.md)
owns the commands, skips and subsequent full forcing results; the
[method record](../docs/contribute/validation/numerics/adaptive-methods.md)
owns the independent numerical references and CPU cloud/droplet results.
Later source changes through `00b1c4f6f2` affect rendering, prose and retained
outputs rather than library execution.

- [x] Full corrected CPU single-cycle and multi-cycle chamber publications.
- [x] Full respiratory publication, preserving its chemical-identity caveats.
- [x] CPU/CUDA wet-to-dry and finite-reservoir comparisons with independent
      references, saved-data replots and downloadable examples.
- [x] Full CUDA forcing publication and its doubled-step/tighter-tolerance
      comparison, with explicit transfer and observation-timing limits.
- [x] Full biomass publication, including all four collision-model branches.
- [x] Full CUDA cyclic chamber, exact second-half continuation, figure review,
      and saved-data replot with the corrected JSON-compatible renderer.
- [x] Full respiratory time-refinement report, including per-species and
      per-particle differences separately from inventory conservation.
- [x] Final source/output consistency review and documentation verification.
- [x] Publish the remaining commits to the existing PR after direct authorization.
- Review readiness is tracked in [PR #1660](https://github.com/uncscode/particula/pull/1660);
  transition from draft requires the preceding gates and hosted checks to pass.

The [follow-through plan](stage-02-cloud-droplet-examples.md) identifies the
completed workloads and acceptance criteria. A completed package test suite
does not substitute for those scientific runs or establish experimental validity.

## Requested fully resident reruns (2026-11-16)

The prior biomass, respiratory-refinement and cyclic-chamber runs were stopped
at the user's request. Their partial output is not completion evidence. The
replacement request preserves full workloads while requiring CUDA
residency and device-owned adaptive decisions, including acceptance/rejection
and step-size selection. The supported solver material profiles have been checked.

- [x] Qualify multi-material GPU RK23/ROS3 rates, surface phases, mappings and
      conservative block solves against independent CPU and dense references.
- [x] Qualify native conditional CUDA adaptation and resident event handoff,
      including exact restart and persistent collision/source RNG.
- [x] Switch biomass AO2008/DNS to supported ST1956 turbulent shear, explicitly
      authorized by the user. Retain dissipation and fluid density, label this
      model change, and preserve all four branches and their full workloads.
- [x] Rerun 10,000-particle biomass preparation and four full-hour branches.
- [x] Rerun the 1,000-particle, twelve-material respiratory publication and full
      20-second coarse/fine comparison, with per-species/particle targets.
- [x] Rerun the 32-box cyclic chamber for 600 seconds and exact 300-second
      continuation, with all existing budgets, observations and forcing.
- [x] Refresh qualified outputs and provenance; run final package/docs gates.

The current resident qualification below supersedes the earlier package gate
for these changes.
The user authorized publication on 2026-11-16; both existing PR branches are
published. Current hosted checks and readiness are recorded in GitHub.

Local resident implementation evidence: multi-material rates and trajectories
cover 3 and 12 exchanging materials, inert and reversed gas mappings, mixed
surface phases, and independent dense block solves. CUDA tests prohibit
physical handoff downloads and host adaptive decisions. Native CPU RK23, ROS2,
and ROS3 retain their existing multi-material support. Exact checkpoint tests
include persistent Brownian, settling and shear RNG streams. Selective chamber
forcing preserves untouched device gas and evaluates humidity on CUDA.

The first resident full-run attempts were superseded during qualification:
the chamber needed selective gas forcing, and serial reservoir sums were too
costly at the full particle counts. Their partial respiratory comparisons and
bounded biomass diagnostics are not full-run completion evidence. Final runs
must use the qualified parallel-reduction revision and retain every original
workload and tolerance, with the explicitly authorized ST1956 model change.

The final resident implementation at `9c548d9d61` passed 10,874 package tests
with 91.02% coverage on actual CUDA. The full biomass and respiratory reruns
and the 201-observation respiratory refinement completed from that same fixed
revision. Their refreshed bundles and compact refinement report retain source
hashes and RTX 5060 identity. The full cyclic trajectory and exact restart passed
at `9c548d9d61`, but plot-data serialization failed afterward. A corrected full
publication completed at `3d3753d2b` with unchanged physics. All 31 frames,
figure inspection and CPU-only replot passed; exact restart, conservation and
zero state transfers between stages remain verified.

## Post-review discrete-bin qualification (2026-11-16)

Stage 2A correction `bd52afd6f4` is integrated into this branch by merge
`18641ea803`. The CUDA trial-domain guard had incorrectly required zero-density
discrete bins to have zero defining masses. It now distinguishes discrete bins
from resolved free slots, whose zero-mass rule remains enforced. Stage 2B extends
the regression across host RK23/ROS3 and native resident CUDA RK23/ROS3, including
heterogeneous three-box trajectories, unchanged empty-bin geometry, independent
species budgets and CPU-reference agreement.

The affected GPU ODE, resident and multi-material tests passed **69 tests** with
**two intentional resident-on-CPU skips**. Full package validation passed **10,884 tests,
23 skips and one expected failure**, with **91.02% coverage** and warnings
as errors in 315.79 seconds on the actual RTX 5060.
Standard Mypy passed **696 sources**; Ruff and format passed **780 files**.
All **89** saved bundles pass source freshness using `scripts/docs_examples.py
check --changed --base HEAD`; no example executable changed in this correction.
Documentation structure reports zero errors.
Commands and logs are retained in `.artifacts/discrete-bin-*.log`; the package
command uses `--cov=particula --cov-fail-under=80 -Werror` without test filtering.

The completed biomass, respiratory and cyclic-chamber workloads all use resolved
particles. Their physical validation behavior is unchanged by this discrete-bin
admission correction. Their retained full-run revisions and hashes remain
`9c548d9d61` and `3d3753d2be`; no bundle is relabeled as executing the later code.
The original scientific results and publication qualification above remain
separate from this post-review package validation.

Plan headers and the published PR descriptions reflect completed full runs and
resident process handoffs. Both branches are published; the Stage 2A remote
review thread is resolved with its regression evidence. Hosted checks on older
heads do not qualify later commits. No merge or release is claimed.

Publication exposed a CI trigger gap: workflows accepted PRs targeting `main`
only, so the dependent 2B PR had no hosted checks. The six existing validation
workflows now also accept PRs targeting `codex/stage-*`; push, release and
deployment triggers are unchanged. The workflow/plugin/feedstock contract suite
passed **93 tests**, and Ruff/format passed for the adjusted existing test.
Hosted validation and review readiness are verified against each published head.
