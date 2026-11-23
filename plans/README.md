# Particula development plans

Reviewed against upstream `main` at `ebabce042e` on 2026-11-19.

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
The first Stage 3A, 3B and 4 deliveries are on main. Stage 5 qualified
closed-box initial-size recovery on CPU and CUDA in
[PR #1676](https://github.com/uncscode/particula/pull/1676).
Stage 6 chemistry is now on main. It covers native gas reactions,
NCAR gas import, resolved bulk reactions, reactive uptake, restart and CUDA
capture for resolved chemistry. [PR #1687](https://github.com/uncscode/particula/pull/1687)
· [Gas evidence](../docs/studies/validation/condensation/gas-chemistry.md)
· [Particle and coupled evidence](../docs/studies/validation/condensation/resolved-chemistry.md)

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

## Evidence available to the next stages

- **Thermodynamics and observations.** Homogeneous binary BAT activity is
  available in kinetic condensation on CPU and CUDA. The citric-acid study
  compares water activity with measurements and a separate AIOMFAC-web
  calculation. The [AIOMFAC project](aiomfac-activity-package.md) supports [native CPU water condensation](aiomfac-cpu-activity.md) and [direct CPU and CUDA activities for fixed and equilibrated mixtures](aiomfac-speciation-and-cuda.md), plus [two-vapor CUDA condensation and moist parcel activation](aiomfac-condensation-integration.md). Kinetic phase separation remains a follow-on. Viscosity is deferred. [BAT qualification](../docs/studies/validation/material-properties/binary-bat.md)
  · [Citric-acid comparison](../docs/studies/literature/particle-water-uptake/citric-acid/index.md)
- **GPU workloads.** Stage 4 measured direct-BAT CPU and tabulated-BAT CUDA
  humidity pulses through 128 boxes, including restart across a vapor reset.
  The finer tested timestep met the radius and water-transfer targets at every
  batch size. [Campaign record](../docs/studies/benchmarks/bat-humidity-pulse/index.md)
- **Observation models.** The CCN retrieval and NaCl chamber studies show how
  instrument selection and observation uncertainty affect the quantity being
  fitted. Stage 5 used synthetic size observations and fixed process
  parameters. [Initial-size recovery and resolution](../docs/studies/validation/condensation/initial-size/index.md)
  · [CCN retrieval](../docs/studies/literature/particle-water-uptake/ccn-retrieval/index.md)
  · [NaCl scavenging](../docs/studies/literature/nacl-scavenging/index.md)
- **Chemistry numerics.** Stage 2A delivered CPU stiff solvers and a Robertson
  assessment. The existing first-order gas chemistry example supplies a
  separate check on native species binding and mass accounting. The Stage 6
  gas milestone now covers resident CUDA and same-backend continuation.
  [Solver assessment](../docs/studies/validation/condensation/adaptive/index.md)
  · [Timed chemistry example](../docs/studies/examples/custom-processes/timed_box_chemistry.py)

## Continuous literature discovery, outside the stages

Twice daily, a discovery run ranks candidate comparisons. An hour later,
a study run resumes interrupted work or selects the highest-ranked eligible
comparison. Each run takes on at most one new study. Selection rotates across
aerosol processes and favors questions that benefit from multiple GPU simulations. Each
study has a fresh worktree and branch. Drafts live in the main checkout's
`.artifacts/literature/` for researcher review. Approved comparisons become
documented examples in individual PRs. Diagnosed Particula gaps go through
issue review. A useful outcome clarifies a published result, tests its reach,
or identifies a follow-up that matters to aerosol researchers.

The [aerosol and cloud data catalog](datasources.md) links published papers to
chamber runs, field measurements and model output for candidate studies.

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
- Keep scientific evidence and executable calculations with their owning
  [studies](../docs/studies/index.md). Keep qualification procedures and release
  gates in the corresponding [study records](../docs/studies/validation/index.md). Plans
  describe pending outcomes rather than accumulating execution logs.
- Follow the [validation reference](../plugins/particula-helper/references/repository-validation.md).
  Missing CUDA or experimental evidence remains missing; do not substitute
  CPU parity or conserved mass for the required evidence.

## Decisions for the next deliveries

1. Select the next Stage 5 inverse problem for composition or hygroscopicity.
   The [Stage 6 delivery](stage-06-chemistry-model.md) records the qualified
   chemistry cases and the physical limits of its uptake law.
2. Choose the literature scheduler's cadence and compute budget at activation.
