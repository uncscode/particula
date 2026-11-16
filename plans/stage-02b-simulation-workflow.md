# Stage 2B: Simulation workflow and reproducibility

Status: core workflow delivered in draft
[PR #1660](https://github.com/uncscode/particula/pull/1660);
[cloud/droplet example follow-through](stage-02-cloud-droplet-examples.md) locally
complete, with later commits awaiting publication authorization.
See the [delivery ledger](stage-02b-delivery.md).
Entry dependency: an initial
[Stage 1 reference case](stage-01-scientific-correctness.md).
Coordinate accepted-step and time semantics with [Stage 2A](stage-02a-numerical-integration.md).

## Execution and delivery

Create a dedicated worktree and `codex/stage-02b-simulation-workflow` branch
when starting this plan. Develop and validate there, then commit, push, and
create a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).

## Outcome

Let a researcher configure explicit physics, run for a physical duration,
inspect time evolution, save selected results, and restart supported runs
without rewriting orchestration and bookkeeping for every study.

The goal is convenience around scientific setup, not a wholesale replacement
of the native containers or hidden selection of physical assumptions.

## Existing foundation

CPU runnable composition and prepared GPU execution already exist. Prepared
simulations provide diagnostics, detached snapshots, and continuation
checkpoints. These are building blocks, not a promised portable disk format.
Saved documentation results support replotting but are not general restart
files. Reuse these capabilities and preserve their ownership boundaries.

## Work packages

1. **Define the research workflow.** Use the shared case to specify duration,
   forcing, output times, selected variables, units, and provenance. Prefer a
   small convenience layer over another object-model redesign. Species-name
   conveniences must expose their resolved maps and reject ambiguity.
2. **Add recording actions.** Support a composable recorder/save action with
   declared observation timing. The scheduler should distinguish observers
   from physics: recording normally occurs after an accepted complete step,
   not at RK stages, rejected trials, or each half of a split step.
3. **Handle physical output times.** Decide whether to land exactly on requested
   times or use validated interpolation for eligible quantities. Define final
   output, coincident forcing events, and resumed runs without duplicate rows.
4. **Provide analysis output.** Save selected diagnostic histories and optional
   particle snapshots with units, dimensions, species order, population measure,
   volume, configuration, source/package identity, and numerical settings.
   Evaluate interoperable formats and compression without choosing one solely
   for convenience. Keep full-state recording opt-in and bounded.
5. **Provide restart output.** Define a versioned restart format carrying native
   state, process configuration, clock, RNG streams, solver/controller state,
   and required auxiliary fields. Reuse qualified GPU checkpoint semantics.
   Advertise exact continuation only for explicitly supported configurations;
   reject incompatible schema, device, or process changes rather than guessing.
6. **Close the example workflow.** Replace the exaggerated first simulation
   with a realistic, plotted, saved time series. Demonstrate loading results,
   continuing a run, and adapting one scientific input while keeping assumptions
   visible. Document migration if public interfaces change.

## Owning code and documentation

- [Runnable composition](../particula/runnable.py),
  [prepared diagnostics](../particula/execution/prepared_diagnostics.py), and
  [prepared checkpoints](../particula/execution/prepared_checkpoint.py).
- [Prepared simulation guide](../docs/guide/gpu/prepared.md) and
  [first CPU simulation](../docs/guide/getting-started/first-simulation.md).
- [Examples](../docs/examples/), [example tests](../examples_tests/), and
  [authoring rules](../website/AUTHORING.md).

## Acceptance and validation

- A realistic CPU example produces a physically timed history, a figure, and
  reloadable output with sufficient information to reproduce the result.
- Eligible CPU and GPU runs resume with the documented continuation guarantee,
  including stochastic state. Analysis-only files cannot masquerade as exact
  restart checkpoints.
- Passive observation does not mutate state, consume physics RNG draws, or
  change the accepted trajectory for a fixed step schedule. If output times
  explicitly shorten steps, document that numerical policy and verify agreement
  within the declared accuracy or stochastic uncertainty, rather than requiring
  identical trajectories. Tests cover rejected trials, split steps, final time,
  resumed schedules, and forcing.
- Incomplete writes cannot replace the last usable checkpoint. Unsupported or
  corrupted files fail clearly; no arbitrary code execution is needed to load
  an analysis result.
- Compression/file writing stays outside CUDA capture; downloads and their
  cost are explicit. Shared storage must not mutate while it is serialized.
- User study output is separate from documentation publication: preserve the
  existing compact saved-data policy and exclude full particle states from
  committed example bundles.

Select focused checks using the
[validation reference](../plugins/particula-helper/references/repository-validation.md).
Storage format, public convenience names, and supported restart combinations
remain design decisions to resolve before implementation.
