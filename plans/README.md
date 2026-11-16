# Particula development plans

Updated: 2026-11-15. Status: proposed development scope for discussion.

These seven plans organize the next development work into five stages. The
previous six plans are completed and retired; they are removed rather than
carried forward as unfinished work. Their history remains in Git.

The aim is a dependable aerosol research framework: scientifically supported
results, practical research workflows, useful acceleration, and reproducible
inference. Stage numbers indicate the intended order of major outcomes, not a
requirement to stop all other work until an entire stage is finished.

## Staged development

| Stage | Plan | Intended outcome |
| --- | --- | --- |
| 1 | [Scientific correctness and verification](stage-01-scientific-correctness.md) | Correct known defects and establish independent reference cases and accuracy boundaries. |
| 2A | [Numerical integration and process splitting](stage-02a-numerical-integration.md) | Supported CPU/GPU stepping choices with measured accuracy, error control, and explicit process ordering. |
| 2B | [Simulation workflow and reproducibility](stage-02b-simulation-workflow.md) | Convenient run, record, save, plot, and restart workflows with explicit scientific inputs. |
| 3A | [Thermodynamics, chemistry, and additional physics](stage-03a-thermodynamics-and-chemistry.md) | First connect existing thermodynamic capabilities; select further physics from demonstrated research needs. |
| 3B | [Scientific and experimental comparisons](stage-03b-scientific-comparisons.md) | Reproduce selected published results and compare predictions with independent measurements. |
| 4 | [GPU scientific usefulness and reliability](stage-04-gpu-qualification.md) | Establish reliable acceleration of complete scientific workloads at comparable accuracy. |
| 5 | [Inverse modeling and automatic differentiation](stage-05-inverse-modeling.md) | Recover supported initial-state quantities with tested inference and gradient methods. |

Stage 2A and 2B can proceed together once a small Stage 1 reference set exists.
Select Stage 3 comparison cases early so they guide both efforts. Stage 3A's
initial thermodynamics integration and Stage 3B's comparisons can also proceed
together; a case using current physics need not wait for new chemistry.

Stage 4 owns the broader GPU workload qualification campaign. CUDA correctness
checks belong in every earlier plan that changes GPU behavior. Stage 5 depends
on a qualified forward workflow for its chosen problem, not on finishing every
possible physics extension or GPU benchmark.

## Continuous literature discovery, outside the stages

The [literature scheduler](literature-scheduler.md) is a separate proposal for
recurring discovery and candidate screening. It feeds Stage 3B and can run
alongside any stage. Writing this proposal does not create an automation, set
its execution time, or authorize automatic reproduction studies.

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
   Broader chemistry and additional physics remain separately scoped follow-ons.
5. Set the literature scheduler's cadence and time when activating it; those
   settings are independent of the development stages.
