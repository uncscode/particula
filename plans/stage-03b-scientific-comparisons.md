# Stage 3B: Scientific and experimental comparisons

Status: proposed. Begin case selection early; quantitative completion depends
on the relevant [Stage 1](stage-01-scientific-correctness.md),
[Stage 2A](stage-02a-numerical-integration.md), and
[Stage 2B](stage-02b-simulation-workflow.md) outcomes.

## Execution and delivery

Create a dedicated worktree and `codex/stage-03b-scientific-comparisons` branch
when starting this plan. Develop and validate there, then commit, push, and
create a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).
Studies selected by the [literature scheduler](literature-scheduler.md) each
use their own iteration worktree and branch rather than this stage's branch.

## Outcome

Demonstrate what Particula can reproduce or explain for a small set of relevant
published studies. Preserve enough inputs, data provenance, code, and uncertainty
information for another researcher to repeat and assess each comparison.

This plan executes selected studies. Recurring paper discovery is managed
separately by the [literature scheduler](literature-scheduler.md), outside the
stage sequence.

## Work packages

1. **Choose the first comparison.** Prefer a well-specified chamber growth/loss
   or organic partitioning case with accessible inputs and outputs. Confirm
   geometry, forcing, concentrations, composition, times, and measurement
   definitions. A famous paper with missing inputs is not automatically the
   strongest first benchmark. Final case selection remains open.
2. **Write the comparison specification.** Identify the exact figure, table,
   dataset, or reference-model output; assumptions shared with Particula;
   unavoidable differences; fixed inputs; fitted quantities; and acceptance
   metrics. Declare what conclusion the comparison can support before fitting.
3. **Establish numerical sufficiency.** Demonstrate temporal, spatial/bin or
   particle-resolution convergence as relevant, and sampling uncertainty.
   Check process-order sensitivity and budgets. Numerical uncertainty should
   be small enough to interpret the selected physical discrepancy.
4. **Map predictions to observations.** Use the measurement's observable and
   sampling convention, including instrument response when it matters.
   Distinguish model number distributions from measured distributions and
   mass-derived quantities from directly observed values.
5. **Compare and diagnose.** Report bias, residual structure, uncertainties,
   and parameter sensitivity. Separate calibration data from independent
   validation. Cross-model agreement establishes agreement under matched
   assumptions, not proof that either model matches nature.
6. **Publish a reproducible example.** Retain source data links and permitted
   compact inputs, configuration, package/source identity, numerical settings,
   plots, and an honest interpretation. Have an independent user try the
   workflow and record reproducibility or usability problems.

## Scope and handoffs

Begin with one bounded study, then broaden to complementary physics. Do not
make a full instrument-analysis platform a prerequisite for a case needing only
one observation transformation. A missing model feature becomes a scoped
[Stage 3A](stage-03a-thermodynamics.md) proposal, or a
[Stage 6 chemistry](stage-06-chemistry-model.md) case, rather than an unrecorded
approximation or automatic feature expansion.

Suitable comparison workflows can later supply
[GPU workloads](stage-04-gpu-qualification.md) and
[inverse problems](stage-05-inverse-modeling.md). Use saved data for plotting
changes; rerun the physical workload when scientific inputs or code change.

## Owning artifacts

- Reusable workflows in [scientific examples](../docs/examples/) with regression
  coverage in [example tests](../examples_tests/).
- Durable findings in [validation records](../docs/contribute/validation/).
- Compact, permitted data under [documentation data](../docs/data/) where
  appropriate; large raw data and full particle states remain outside Git.
- [Documentation authoring](../website/AUTHORING.md) and
  [validation guidance](../plugins/particula-helper/references/repository-validation.md).

## Acceptance and validation

- At least one selected study has a reproducible quantitative comparison with
  declared assumptions, inputs, uncertainty, and numerical convergence evidence.
- The documented claim matches the result: a discrepancy or failed reproduction
  is useful evidence and must not be relabeled as successful physical validation.
- Data access, licensing, conversions, and any digitization are traceable.
  Digitized measurements retain extraction uncertainty and are identified as such.
- An independent user can reproduce the principal figure or metric from the
  supplied workflow without undocumented maintainer intervention.
- No benchmark is tuned against its validation target and then presented as
  independent validation. Remaining physical and observation-model limitations
  are visible beside the comparison.
