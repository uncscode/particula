# Stage 3B: Scientific and experimental comparisons

Status: the aqueous citric-acid study at 293.15 K, its kinetic checks, and the downloadable example are complete on the study branch. The independent model comparison is complete using McGill AIOMFAC-web 3.15. The
[comparison specification](../docs/studies/literature/particle-water-uptake/citric-acid/index.md)
records the target data, fixed inputs, metrics and numerical checks before evaluation.
Stages 1, 2A and 2B are merged. Reuse their verified processes,
numerical controls and reproducibility workflow, qualifying the actual study
configuration before interpreting agreement with measurements.

## Execution and delivery

Create a dedicated worktree and `codex/stage-03b-scientific-comparisons` branch
when starting the selected study. Develop and validate there, then commit,
push, and create a PR when the plan's acceptance criteria are met. Follow the shared
[worktree, branch, and PR workflow](README.md#worktree-branch-and-pr-workflow).
Scheduled literature studies use the same scientific criteria. Each new
comparison has its own worktree and branch. The agent keeps the draft in the main
checkout's `.artifacts/literature/<iteration-id>/` for researcher review.
An approved study becomes a documented example in its own PR. The independent
AIOMFAC calculation below belongs to the citric-acid case; each later study
declares the reference appropriate to its question.

## Outcome

Deliver one reproducible quantitative comparison with a published study,
including declared assumptions, traceable inputs and an honest assessment of
agreement or disagreement. Additional studies are later increments. Preserve
enough data provenance, code and uncertainty information for another researcher
to repeat and assess the comparison.

Stage 3B owns study selection, observation mapping and scientific interpretation.
[Stage 3A](stage-03a-thermodynamics.md) owns any needed native thermodynamic
coupling. Screening can start immediately; a study using current supported
physics can proceed without 3A. A study needing new coupling waits for that
specific qualified capability, not every proposed thermodynamics extension.

Recurring paper discovery and selection follow the
[literature workflow](README.md#continuous-literature-discovery-outside-the-stages).

## First deliverable: select and specify a case

Screen a small shortlist before committing to a full simulation. Chamber
growth/loss and organic/water partitioning are candidate categories, not chosen
papers. Existing biomass, respiratory and cyclic-chamber examples establish
workflows and numerical evidence; they are not automatically independent
experimental validation datasets.

For each candidate, record the primary paper/data link, exact target figure or
table, accessible inputs, missing information, redistribution permissions,
required physics and estimated execution cost. Select the case with sufficient
evidence for a meaningful comparison. If critical inputs are absent, obtain
them or reject the candidate; do not invent them. A shortlist or an inaccessible
dataset does not satisfy the final comparison acceptance criteria.

Before production runs, retain the selected specification with the study's
validation record:

| Item | Required information |
| --- | --- |
| Scientific question | Quantity or process response of interest, proposed explanation and claim the comparison could support. |
| Reference or baseline | Primary source and target, such as a measured series, versioned model output, analytical solution or specified baseline calculation for a process study. Record relevant uncertainties. |
| Inputs | Species identities, composition, concentration basis, geometry/volume, size distribution, temperature, humidity, forcing and times; units and conversions. |
| Model boundary | Processes, activity/phase assumptions, wall losses and reservoirs; known differences and excluded mechanisms. |
| Comparable quantities | Match units, population weighting and time basis. For measurements, include instrument response where material. For a model or process study, define the output and the assumptions shared or varied between cases. |
| Metrics | Absolute/relative error definitions, uncertainty treatment, zero-value handling, and thresholds set before evaluating the result. |
| Calibration | Fixed and fitted inputs, fitting bounds/data, and held-out validation data; label a fit as calibration when independent data are unavailable. |
| Numerical evidence | Solver/tolerances, timestep and population/bin refinement, process order, random seeds/ensemble size where relevant, and budget residual limits. |
| Execution and provenance | Source revision, environment, backend/device, data retrieval and preprocessing, run/replot commands, output location and resource budget. |

Separate requirements for a credible comparison from its scientific result:
missing convergence evidence is unfinished work; a converged disagreement with
measurements is a result to investigate and report. Numerical-error targets must
be small enough to interpret the declared observation uncertainty or effect.

## Work packages

1. **Select and freeze the comparison specification.** Complete the screening
   and specification above. Deliver one selected target with adequate inputs
   and an explicit 3A dependency, if any. Changes after seeing results require
   a recorded reason and an updated distinction between exploration and
   independent validation.
2. **Build the reproducible input and forward workflow.** Retain permitted raw
   or compact source data, extraction/conversion steps, native species maps,
   forcing and budgets. Use Stage 2B recording and supported restart rather
   than introducing a separate simulation interface. Deliver a small regression
   case plus the full scientific run configuration; the small run does
   not replace full-workload evidence.
3. **Establish numerical sufficiency.** Demonstrate temporal, spatial/bin or
   particle-resolution convergence as relevant, and sampling uncertainty.
   Check process-order sensitivity and budgets. Numerical uncertainty should
   be small enough to interpret the selected physical discrepancy.
4. **Compare equivalent quantities.** For measurements, use the instrument's
   observable and sampling convention. For model comparisons, align the
   definitions, inputs and evaluation times. For a process study, quantify
   the response to the declared change against its baseline.
5. **Compare and diagnose.** Report bias, residual structure, uncertainties,
   and parameter sensitivity. Separate calibration data from independent
   validation. Cross-model agreement establishes agreement under matched
   assumptions, not proof that either model matches nature.
6. **Publish a reproducible example.** Retain source data links and permitted
   compact inputs, configuration, package/source identity, numerical settings,
   plots, and an honest interpretation. Verify the example from a clean installation and compare its binary activities
   with McGill AIOMFAC-web at the measured compositions. Record the model
   version, inputs and residuals.

## Scope and handoffs

Begin with one bounded study, then broaden to complementary physics. Do not
make a full instrument-analysis platform a prerequisite for a case needing only
one observation transformation. A missing model feature becomes a scoped
[Stage 3A](stage-03a-thermodynamics.md) proposal, or a
[Stage 6 chemistry](stage-06-chemistry-model.md) case, rather than an unrecorded
approximation or automatic feature expansion.

Use CPU as the reference backend where supported. Add CUDA only when it is part
of the selected study's declared delivery; then qualify equivalent scientific
settings and full outputs on actual CUDA. Stage 4 owns broader speedup claims.
AIOMFAC is a separate follow-on to Stage 3A, not a prerequisite for a comparison
that existing models can support. Chemistry remains Stage 6. This revision
does not activate the literature scheduler.

Suitable comparison workflows can later supply
[GPU workloads](stage-04-gpu-qualification.md) and
[inverse problems](stage-05-inverse-modeling.md). Use saved data for plotting
changes; rerun the physical workload when scientific inputs or code change.

## Owning artifacts

- Reusable workflows in [scientific examples](../docs/examples/) with regression
  coverage in [example tests](../examples_tests/).
- Durable findings in [validation records](../docs/studies/validation/index.md).
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
- The documented clean-install workflow reproduces the principal metric and
  the external AIOMFAC calculation supplies an independent model comparison.
- No benchmark is tuned against its validation target and then presented as
  independent validation. Remaining physical and observation-model limitations
  are visible beside the comparison.
- The full selected workload, numerical refinements and any required ensembles
  are completed. Saved outputs identify their actual source revision; replotting
  does not relabel older simulations as newly executed evidence.
- The example's regression checks, full execution, figure inspection and
  documentation checks pass according to the linked validation guidance.

## Completion and next handoff

Complete the first milestone with one selected study's specification, traceable
data processing, executable example, compact results, quantitative error and
uncertainty report, convergence evidence and the external model comparison, delivered
in its PR. Report whether the physical hypothesis is supported, contradicted or
unresolved; implementation success alone does not establish physical validity.
Missing required evidence remains an open gate.

Identify the qualified forward configuration and observable suitable for Stage 4
GPU qualification and Stage 5 inference. List discovered physics gaps as
separate proposals. No release/version change or additional study campaign is
required to finish this bounded milestone.
