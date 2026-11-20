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
- Start each scheduled run in a **fresh project chat**. Read the shared
  register for topic history, capacity and researcher decisions. If a study
  was interrupted, resume from its packet under its existing ID. Otherwise
  choose **at most one new study** for this chat and queue other promising
  papers for later runs. Deliver the review link in the run's own chat.
- For each new selection, choose a different paper and scientific question
  in a different process family from the last new study. Search across
  underexplored families first. Check recent and pending register entries
  for repeated observables and experiments, including older entries whose
  topic you can infer from their comparison target. Changing a parameter in
  the same experiment belongs in that study. Record the selected family,
  observable and reason for choosing it in the register.
- Use the [aerosol and cloud data catalog](datasources.md) to find candidate
  papers by process or observable. Follow the paper to the specific run,
  flight, station interval or data table. Check the units, any corrections
  applied by the authors, and whether the file contains the quantity the
  proposed study would compare. Continue the wider search when needed.
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

### Choosing a paper for a GPU simulation campaign

Look for papers with a measured response across conditions, documented inputs,
and an observable that could distinguish a simpler model from a more detailed
one. Shortlist across process families. Give priority to a paper for which
many simulations can resolve a physical question. For example, a chamber size
distribution measured over time at several initial concentrations could test
whether including particle-resolved coagulation alongside wall loss improves
the predicted decay. Match the paper's wall-loss treatment and identify which
additional process is the Particula hypothesis.

For each promising campaign, state what the reference calculation predicts,
which added physics or resolution might change it, and which observations
could distinguish the predictions. Check the exact configuration against the
[GPU support guide](../docs/guide/gpu/support.md). Pilot a reference case and
one higher-fidelity case, then use their runtime and memory to choose the
campaign size within the local budget. Batch independent conditions on CUDA,
add independent seeds when sampling variability matters, and refine timestep
or particle count where a numerical shift could obscure the physical effect.
Record the device, completed cases and spread alongside the change in the
observable. An unavailable CUDA run stays pending. If a different paper with
a simpler comparison has stronger evidence, explain the choice in the packet.

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
   the outstanding check. Write the packet's [researcher-facing review](#writing-reviewmd)
   in `review.md` and link it from the handoff.
3. **Review the science with the researcher.** Deliver the packet path and a
   short discussion of what was tested, what the evidence supports and what
   would resolve the main uncertainty. Record the researcher's decision to
   publish, revise or archive. A revision resumes the same study. A measured
   disagreement can merit publication, and an inconclusive comparison can
   still identify the next experiment. Pending drafts stay in `.artifacts/`.
4. **Publish the approved comparison.** Publish the reviewed report and its script
   together under [Literature analyses](../docs/studies/literature/index.md).
   Keep the observable, comparison, provenance and limitations with the report.
   For a table-of-contents figure, pair the report title with a square graphic
   using [particula-toc](../plugins/particula-helper/skills/particula-toc/SKILL.md).
   Link its run key, topics and supporting evidence from the canonical study
   record. Put compact, redistributable inputs in [docs/data/](../docs/data/).
   Register the example, refresh its saved results in `website/results/`, and
   verify that the documented workflow reproduces the reviewed principal metric
   and figure. Follow the [example authoring workflow](../website/AUTHORING.md).
   After validation and diff review, open one PR for that study and record its
   URL. Material changes to the scientific result return to research review.

## Writing `review.md`

Write a short technical note for a researcher who knows aerosol science and
has not read the paper. Give each review one scientific question. Open with
what the authors measured or modeled, what Particula compared, and what the
result means for someone designing an experiment or interpreting data. If
the evidence leaves the question open, say what remains unresolved and why
it matters.

Use [particula-studies-literature](../plugins/particula-helper/skills/particula-studies-literature/SKILL.md) for the analysis and write-up. Answer the study question in the title. A measured contrast, its direction and the condition where it appears give readers a reason to open the report. Carry that claim through the section headings. For example,
“Particula assigns 102 nm to a 100.7 nm dry PSL standard near atmospheric
pressure” names the quantity, reference and conditions. In the text, spell
out polystyrene latex (PSL) and explain how electrical mobility is converted
to diameter. The +1.3% is a shift in **inferred dry diameter** relative to
the reference sphere. A claim about hygroscopicity would require a separate
comparison. Define symbols where they first affect the interpretation, such
as the Knudsen number (Kn), the ratio of gas mean free path to particle radius.

Build the narrative around the evidence and the decision it supports.

- **Put a number in context.** Give the reference and comparison values,
  units, direction and conditions for the main result. For a percentage,
  identify the denominator. Use “uncertainty” for a quantified uncertainty
  estimate and “shift” or “difference” for a change between calculations.
- **Explain when the effect matters.** Show where it is small or large
  relative to a stated measurement uncertainty or experimental tolerance,
  when available. Explain the physical reason if known. Identify which
  assumptions came from the authors and which the Particula analysis added.
  For example, in a chamber study, say whether the authors used size-dependent
  wall loss, a constant rate, or left the treatment unspecified. Explain how
  that limits what particle decay can tell the reader about collisions.
- **Label the evidence where it appears.** Distinguish observations,
  quantities inferred from observations, and synthetic predictions in the
  prose and figure captions. Use a table or figure when it makes a comparison
  easier to read. Its caption should identify the observable, conditions and
  reference that give the pattern meaning.
- **Close with a research decision.** Say what the reader can do with the
  result, where it applies, and which next measurement or calculation could
  resolve the main uncertainty. Put reproduction commands and supporting
  files at the end. Link the paper's specific table, figure or equation
  beside the claim it supports.

Before handing off, read just the title, headings and first paragraph. A
researcher skimming those lines should be able to say what changed, under
which conditions, and why it matters.

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

The [citric-acid comparison](../docs/studies/literature/particle-water-uptake/citric-acid/index.md)
illustrates this tone. “Marsh and colleagues reported uncertainties in both
composition and water activity, which let us test the model across all 12
measured compositions. Our BAT calculation gives higher water activity at
each point. Measurements at matched compositions across temperatures could
help establish how much temperature contributes to that difference and where
the model is useful for predicting droplet water uptake.”
