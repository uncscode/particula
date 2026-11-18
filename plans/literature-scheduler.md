# Literature scheduler for independent aerosol research

Status: proposed. Choose the cadence at activation.

## Scientific purpose

An automated research agent uses Particula to conduct independent aerosol
comparisons and support researchers in extending and clarifying published
results. Choose questions whose answer matters to aerosol science. A comparison
might establish where a published relationship holds, clarify which assumption
controls a prediction, or identify a measurement or model capability that
would distinguish two explanations. Record what the agent and other researchers
can test next and why the answer would matter.

Search primary papers, supplements, author-maintained repositories and archived
model inputs. Include older reference studies. Initial topics include
condensation and evaporation, coagulation, chamber growth and loss, charged
particles and organic/water thermodynamics. Follow the scientific comparison
requirements in [Stage 3B](stage-03b-scientific-comparisons.md).

Experimental comparisons, model intercomparisons and process studies are all
eligible. A target can be a measured distribution, another model's published
trajectory, an analytical scaling, or the change between two specified process
configurations. For example, varying coagulation while holding wall loss and
initial conditions fixed can show when coagulation affects a predicted chamber
size distribution. That comparison quantifies sensitivity within the model.

## Scheduling and selection

- Set the interval according to how long studies and researcher review take.
  Invocations may be weekly or more frequent when capacity permits. At
  activation, choose the timing in America/Denver, the delivery destination,
  and the per-study and total compute
  and storage budgets. Use the application's automation controls, check for an
  existing schedule, and verify the saved settings.
- Each invocation can select **zero to three new comparisons**. Resume
  interrupted work first and consider work already awaiting review.
  Queue additional candidates when capacity or resources are committed. An
  open PR keeps its study identity for follow-up. Never start a second copy.
- Report up to three strong candidates. For each, identify the paper and data
  sources, specific comparison target, original contribution, proposed
  question, scientific value, available inputs and uncertainties, reuse terms,
  supported physics and backend, and expected effort. Name the decision the
  result could inform. If none qualifies, record why and close the search.
- Select comparisons that the current Particula checkout already supports.
  Verify the process, domain, representation and quantity to compare for that
  target. A supported subset of a paper can answer a worthwhile question in
  its own right. Record missing study inputs separately from gaps in
  Particula. Revisit deferred candidates when data or model support change.

## Shared record and draft storage

Keep the study register and review packets under `.artifacts/literature/` in
**the persistent main checkout**. Resolve this shared directory once at
activation and give every study worktree its absolute path. A worktree's own
`.artifacts/` is a different directory. Configure storage and retention at
activation so a pending review survives worktree cleanup and interruptions.

Identify a comparison by DOI or stable identifier **and comparison target**.
Record a unique iteration ID, status, study directory, worktree, branch, base
and execution revisions, review decision, and any issue or PR link. Track
queued, active, awaiting review, blocked, approved for publication and closed
outcomes. Record the reason for a deferral or closure. Use one writer or an
atomic claim to reserve the comparison and capacity together before creating
a worktree. On restart, reconcile the register with existing worktrees and PRs.

## From comparison to reviewed example

1. **Specify the comparison.** Identify the reference or baseline. For
   measurements, match the sampling and instrument conventions. For model
   intercomparisons, match inputs and output definitions and document differing
   assumptions. For process studies, specify what varies, what stays fixed,
   and which response would distinguish the cases. Define units,
   fixed and fitted inputs, metrics, relevant uncertainties and
   numerical refinement before evaluating the target. A proposed extension
   gets its own question and is labeled as a new prediction. Follow the
   [worktree and branch workflow](README.md#worktree-branch-and-pr-workflow)
   for each new study, with a fresh worktree and branch such as
   `codex/literature-<timestamp>-<paper-slug>`. Resume that same iteration
   after interruption.
2. **Run and retain a review packet.** Run against the study checkout and keep
   the draft percent-format Python script, source and data provenance, inputs
   or retrieval instructions, run/replot commands, versions, compact numerical
   results, figure and interpretation in `.artifacts/literature/<iteration-id>/`
   in the shared location. Record refinement, uncertainty and applicable
   budgets. Keep bulky or restricted data in suitable storage and record their
   location and reuse conditions. If a budget is exhausted, save progress and
   the outstanding check.
3. **Review the science with the researcher.** Deliver the packet path and a
   short discussion of what was tested, what the evidence supports and what
   would resolve the main uncertainty. Record the researcher's decision to
   publish, revise or archive. A revision resumes the same study. A measured
   disagreement can merit publication, and an inconclusive comparison can
   still identify the next experiment. Pending drafts stay in `.artifacts/`.
4. **Publish the approved comparison.** Promote the reviewed script into the
   relevant topic under [docs/examples/](../docs/examples/). Put durable
   evidence and limitations in [docs/contribute/validation/](../docs/contribute/validation/)
   and any compact, redistributable inputs in [docs/data/](../docs/data/).
   Register the example, refresh its saved results in `website/results/`, and
   verify that the documented workflow reproduces the reviewed principal metric
   and figure. Follow the [example authoring workflow](../website/AUTHORING.md).
   After validation and diff review, open one PR for that study and record its
   URL. Material changes to the scientific result return to research review.

## Diagnosing a Particula limitation

When a result is unexpected, check units, inputs, measurement conventions,
model assumptions, numerical convergence and the version-matched implementation.
Reduce a suspected implementation problem to a small shareable case when
possible. Compare expected and observed behavior, state the research impact,
and distinguish a defect, an unsupported capability and an unresolved physical
discrepancy. A residual against a paper alone does not diagnose a software bug.

For a diagnosed Particula defect or capability gap, follow the
[issue reporting workflow](../plugins/particula-helper/skills/particula-report/SKILL.md).
Search existing issues and fixes, show the researcher the complete proposed
report, then file the reviewed issue or add the approved evidence to an
existing issue. Link it in the study record. Pause the affected claim until
the required capability is available. Independent supported comparisons can
continue. Library implementation belongs to the follow-up issue.

## Communicating the contribution

Identify the work as an automated, independent Particula analysis and credit the
paper, data and methods. In reports, examples, PRs and issue drafts, lead with
what the authors made possible. Describe the observable, conditions, size and
uncertainty of the effect. Attribute choices to the team that made them, name
plausible explanations as hypotheses, and identify one concrete next test and
who could use its result. State what received human review when applicable.

The [citric-acid comparison](../docs/contribute/validation/thermodynamics/citric-acid-comparison.md)
illustrates this tone. “Marsh and colleagues reported uncertainties in both
composition and water activity, which let us test the model across all 12
measured compositions. Our BAT calculation gives higher water activity at
each point. Measurements at matched compositions across temperatures could
help establish how much temperature contributes to that difference and where
the model is useful for predicting droplet water uptake.”
