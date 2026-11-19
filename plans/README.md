# Particula development plans

Updated: 2026-11-17. Status: staged development and delivery tracking.

These eight plans organize the next development work into six stages. The
previous six plans are completed and retired; they are removed rather than
carried forward as unfinished work. Their history remains in Git.

The aim is a dependable aerosol research framework: scientifically supported
results, practical research workflows, useful acceleration, and reproducible
inference. Stage numbers indicate the intended order of major outcomes, not a
requirement to stop all other work until an entire stage is finished.

Stages 1, 2A and 2B are merged through
[PR #1658](https://github.com/uncscode/particula/pull/1658),
[PR #1659](https://github.com/uncscode/particula/pull/1659), and
[PR #1660](https://github.com/uncscode/particula/pull/1660), respectively, with
[cloud/droplet integration examples](stage-02-cloud-droplet-examples.md) now
qualified and published, including the full resident CUDA reruns.
Stage 3A has a qualified homogeneous BAT implementation on its delivery branch. Stage 3B covers one published-study comparison. Stage 4 qualified the [BAT humidity-pulse ensemble](../docs/contribute/validation/gpu/bat-humidity-pulse.md) through 128 boxes. Stages 5–6 remain proposed.
AIOMFAC activity is a separate follow-on; viscosity remains deferred.

## Staged development

| Stage | Plan | Intended outcome |
| --- | --- | --- |
| 1 | [Scientific correctness and verification](stage-01-scientific-correctness.md) | Correct known defects and establish independent reference cases and accuracy boundaries. |
| 2A | [Numerical integration and process splitting](stage-02a-numerical-integration.md) | Supported CPU/GPU stepping choices with measured accuracy, error control, and explicit process ordering. |
| 2B | [Simulation workflow and reproducibility](stage-02b-simulation-workflow.md) | Convenient run, record, save, plot, and restart workflows with explicit scientific inputs. |
| 3A | [Thermodynamics and additional physics](stage-03a-thermodynamics.md) | Add binary BAT activities to kinetic condensation on CPU and resident GPU, starting with homogeneous droplets. |
| 3B | [Scientific and experimental comparisons](stage-03b-scientific-comparisons.md) | Reproduce and extend published results through experimental comparisons, model intercomparisons and process studies. |
| 4 | [GPU scientific usefulness and reliability](stage-04-gpu-qualification.md) | Establish reliable acceleration of complete scientific workloads at comparable accuracy. |
| 5 | [Inverse modeling and automatic differentiation](stage-05-inverse-modeling.md) | Recover supported initial-state quantities with tested inference and gradient methods. |
| 6 | [Configurable chemistry model](stage-06-chemistry-model.md) | Bind reactions to user species with shared CPU/GPU configuration, then qualify gas, particle, and coupled chemistry. |

The [Stage 3A validation record](../docs/contribute/validation/thermodynamics/bat-activity.md)
tracks direct BAT verification, CPU/CUDA transfer, capture, restart and phase
limits. Stage 3B owns comparison data, observation mapping and independent
reproduction. Its separate branch carries the citric-acid study.

Stage 4 owns the broader GPU workload qualification campaign. CUDA correctness
checks belong in every earlier plan that changes GPU behavior. Stage 5 depends
on a qualified forward workflow for its chosen problem, not on finishing every
possible physics extension or GPU benchmark.

Stage 2A begins with an assessment of error estimation, higher-order methods,
stiff methods, and conservative alternatives before selecting production
solvers. Its small chemistry benchmark informs Stage 6 without bringing the
full chemistry feature into the early stages.

Stage 6 is a later development priority. Its technical prerequisites are
verified state/accounting, suitable numerical methods, recording/restart, and
the thermodynamics required by the selected mechanism. It does not depend on
finishing autodiff; chemistry gradients are outside its initial scope.

## Continuous literature discovery, outside the stages

The proposed [literature scheduler](literature-scheduler.md) uses an automated
agent to conduct independent aerosol research alongside the stages. Each
run starts in a fresh project chat, shortlists up to three papers, and takes
on at most one new study. Selection rotates across aerosol processes and favors
questions that benefit from multiple high-fidelity GPU simulations. Each
study has a fresh worktree and branch. Drafts live in the main checkout's
`.artifacts/literature/` for researcher review. Approved comparisons become
documented examples in individual PRs. Diagnosed Particula gaps go through
issue review. A useful outcome clarifies a published result, tests its reach,
or identifies a follow-up that matters to aerosol researchers.

## Worktree, branch, and PR workflow

Apply this workflow whenever executing any plan in this folder:

Before implementation, record a bounded completion target: the selected case,
included work packages, deliverables, acceptance metrics, supported backends,
and deferred extensions. Resolve the open decisions needed for that target.
Continuous verification and optional future physics do not make a plan's first
delivery unfinishable. A smaller milestone must be labeled explicitly and must
not claim completion of omitted acceptance criteria.

1. **Create an isolated worktree and branch.** Before implementation, inspect
   existing worktrees, branches, remotes, and open PRs. Fetch and verify the
   intended upstream base; `origin` may be a fork. Create a dedicated Git
   worktree and a new `codex/<plan-slug>` branch for that plan. Record the
   worktree path, branch, and base revision in the execution report. Preserve
   unrelated changes in the starting checkout. If prerequisites are still on
   another branch, identify that dependency and the intended PR base explicitly.
2. **Develop inside that worktree.** Read its current repository guidance and
   carry out the plan's work packages, scientific acceptance criteria, and
   relevant documentation changes on its dedicated branch. Use that checkout's
   environment; do not accidentally import or test another worktree's code.
3. **Validate and review.** Run the checks required by the changed behavior,
   inspect the complete diff, and fix findings. Record commands, results,
   scientific evidence, and remaining limitations. A missing required check
   remains an incomplete requirement rather than a passing result.
4. **Commit and open a PR when complete.** Commit the intended changes, push
   the plan branch to the appropriate writable remote, and create a PR against
   the verified target repository/base. Follow its PR template and describe the
   outcome, validation, and limitations. Include the PR URL in the completion
   report. PR creation is part of executing the plan; merging is a separate step.
5. **Resume without duplicating work.** An interrupted execution resumes its
   existing worktree and branch. Update an existing PR for the same work rather
   than opening a duplicate. Keep unfinished work available for continuation.
   After merge, record the merged revision and carry durable evidence forward.
   Remove a worktree only after confirming its work is preserved and it has no
   uncommitted changes or study outputs that still need retaining.

Each staged plan owns its own worktree and branch. The literature scheduler
uses this same workflow separately for **each new selected study iteration**;
it must not accumulate different studies on one long-lived scheduler branch.

## Shared working agreements

- Use the same selected reference cases across corrections, solver development,
  workflow examples, CPU/GPU comparisons, and inference where appropriate.
- Separate implementation verification, numerical convergence, physical
  validation, stochastic uncertainty, and performance evidence.
- Retain explicit units, species maps, population measures, volume accounting,
  supported backends, and source/sink budgets.
- Preserve established state ownership and failure boundaries. New public APIs,
  checkpoint formats, and numerical policies need an explicit compatibility
  decision; proposed names in a plan are not approved interfaces.
- Keep durable results in [validation records](../docs/contribute/validation/)
  and executable examples in the maintained documentation workflow. Plans
  describe pending outcomes rather than accumulating execution logs.
- Follow the [validation reference](../plugins/particula-helper/references/repository-validation.md).
  Missing CUDA or experimental evidence remains missing; do not substitute
  CPU parity or conserved mass for the required evidence.

## Decisions to settle during refinement

1. Select the first shared case: chamber growth/loss, organic partitioning, or
   another case with sufficiently specified inputs and an independent reference.
2. Choose initial error tolerances and observable quantities for that case
   before choosing a default solver or reporting a speedup.
3. Agree on analysis-output and exact-restart requirements before committing to
   a storage format or a new simulation interface.
4. Choose the first thermodynamic connection needed by the selected case.
   Refine Stage 6's species binding and reaction schema separately from the
   solver choice; choose one small mechanism for its first qualification case.
5. Set the literature scheduler's cadence and time when activating it; those
   settings are independent of the development stages.
