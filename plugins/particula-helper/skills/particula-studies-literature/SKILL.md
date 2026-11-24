---
name: particula-studies-literature
description: "Reproduce a published aerosol comparison and extend it with existing data and controlled Particula simulations. Use for process discovery, uncertainty analysis and finding-led literature reports."
---

# Analyze aerosol studies and write literature reports

Use the [drafting and editing roles](../../references/drafting-and-editing.md)
when framing a study question or writing its report. An independently launched
literature analysis can originate the question. The study owner can select
it from the primary evidence and proceed without an intake interview.

## From the paper to a new physical question

1. **Establish what the paper showed.** Choose a claim and its figure, table or equation. Identify what the authors measured, inferred, fitted and predicted. Record the data release and the conditions over which the claim was tested.
2. **Repeat the published comparison.** Start with the authors' inputs and observation conventions. Align time zero, size bins, units, sampling windows and detection limits. Where the paper supplies both observations and a model curve, show both alongside the Particula reference over their common coverage. Check a named feature numerically against the published value or extraction tolerance. Say whether you reran the authors' model, recovered its archived output, or digitized its curve. When only observations or theory are available, name the benchmark you could repeat and the part that remains unverified.
3. **Design an extension from the result.** Identify a gap the paper leaves open, even when its comparison reproduces well. Examples include an untested regime, a process the authors held fixed, two explanations consistent with the same observation, or a bias with a plausible correction. For competing explanations, state what each predicts and which existing observable could distinguish them. For an untested regime, specify the range to explore and the response that would mark a limit of the published result. Hold forcing, particle supply and instrument mapping common when isolating a growth law, for example. Follow the response across the available record and show where an apparent advantage weakens. Explain any restricted window or threshold.
4. **Test whether the evidence distinguishes the processes.** Compare the physical effect with data uncertainty, numerical refinement and plausible alternative assumptions. Where samples share a stage, event or run, preserve that dependence when estimating uncertainty. For a fitted mechanism, record what informed parameter or model selection, freeze the fit and score any reserved observations separately. An already inspected period can test transfer within a run. Predictive superiority across experiments needs comparable independent evidence. If mechanisms remain tied, give their common prediction and the condition where they disagree most.

When proposing a solution, define the problem and the success criterion before tuning. Compare the proposed change with the original treatment under matched conditions. Quantify what improves, what worsens, and where the gain survives uncertainty and refinement. State which outcomes observations support and which are predictions for an unobserved regime. A useful result can also explain why a proposed correction fails.

Let the question determine the simulation effort. Use paired runs for a single process change. When mechanisms interact, use [chamber analysis](../particula-chamber-analysis/SKILL.md) for bounded fits, ensembles or particle-resolved replay, and [reaction discovery](../particula-reaction-discovery/SKILL.md) for competing chemical pathways. A synthetic record with known mechanisms can test what a fit recovers. Evidence for a mechanism in the experiment comes from the observed record.

For these literature studies, follow up by reanalyzing an existing record, retrieving an accessible published dataset, or running a specified simulation. Give the data field or model change, observable and outcome that would change the interpretation. If an unmeasured quantity prevents attribution, state that limit. A simulation can bound its consequences.

## Run a self-directed research loop

For a scheduled literature study, plan **three hours of active simulation and analysis** on the selected comparison. Start from the paper and the reproduced baseline. Set your own study goal in the packet. Name the physical question, observable, comparison and evidence that would support a conclusion, including an inconclusive one. For example, “Determine whether changing particle arrival times can account for the delayed count rise while preserving the measured spectrum.” Keep that goal visible as the individual tests change.

1. **Predict.** Use the last result to choose the uncertainty that matters most. Before the next run, record the hypothesis, its alternative and the result that would distinguish them. Pick the smallest informative simulation or analysis of existing data.
2. **Run and inspect.** Execute the comparison, link its inputs and saved output, and check the observable in physical units. Inspect where the change helped, had little effect or worsened agreement. Compare the difference with the relevant uncertainty.
3. **Choose the next test.** If the effect is clear, challenge it with refinement, a second diagnostic or another observed regime. If two explanations remain plausible, choose a condition where their predictions separate. If a hypothesis fails, retain the result and use it to narrow the next question. Keep reserved observations out of this adaptive search until the cases and settings are frozen.
4. **Checkpoint and repeat.** After each cycle, save a short entry with the prediction, completed comparison, quantitative result, interpretation, elapsed research time and next decision. Begin the next feasible test directly. On resumption, read the last entry and carry forward completed runs and time used.

Use the three hours while informative work remains. A first successful fit leaves room to challenge its explanation. If the goal has been answered and challenged, or the available data and supported simulations cannot distinguish the remaining explanations, record the evidence for stopping early. At the time target, verify the strongest claim and queue the highest-value unfinished test under the study's outer limits. Summarize the decisions that changed the interpretation in the report.

## Analyze the results

1. Read run metadata and numerical outputs before interpreting plots. Establish units, distribution representation, radius/diameter convention, species and process configuration.
2. Check NaN/Inf, negative concentrations/masses, number and mass trends, and the correct open/closed-system inventory. Condensation exchanges gas/particle mass; coagulation changes number while conserving mass in the ideal closed model. Boundary losses or sources require explicit budgets.
3. Distinguish per-bin counts, PDFs and dN/dlogD; integrate with appropriate widths and Jacobians. Never sum a continuous density as though it were per-bin counts.
4. Compare smaller time steps and finer resolution for numerical convergence. Report changes in physically relevant observables, not only plot appearance. For stochastic studies use independent seeds and uncertainty intervals when available.
5. Label axes and units; use logarithmic size axes where appropriate. Preserve numerical observations and export plots plus a short report of inputs, versions, checks and limitations. Reuse saved data for plotting-only changes; do not rerun physics or reconstruct values from images. Mark sampled observations explicitly rather than implying exact event times. Registered repository examples follow `../particula-docs/SKILL.md` for stable folders, bounded plot data and replot provenance.
6. For observations align units, sampling windows, instrument response and uncertainties; do not conflate calibration and independent validation. Separate numerical, parameter and structural uncertainty.
Deliver key findings, linked plots/data, quantitative check results and justified next steps. A successful bundled smoke test is not evidence that a model predicts a particular experiment.

## Build the publication draft from the research loop

Write the packet's `review.md` as the manuscript another aerosol researcher would read. If the packet already has a main analysis, use that manuscript as the publication draft and link to it from `review.md`. Maintain one scientific narrative across revisions.

Read the cycle results together. Select the claim that changes how a reader would interpret the published result, choose a model, or design a simulation. Show the matched reference and the Particula extension in a figure, quantify the effect and its uncertainty, and bring in the strongest counterexample or competing explanation. Include a negative result when it rules out a tempting explanation or sets a useful bound. Move detailed runs to linked supporting pages and introduce each through the finding it contributes.

Make the value to the reader concrete. For example, if two wall-loss fits reproduce the same dark decay yet predict different growth times, state which forecast remains uncertain and show how to run both fits against another archived record. Close with the conditions under which the conclusion applies and one specific way to build on it. Link the original data, compact plot data and replay instructions. Verify the central figure and claim before handing the draft to the researcher for the publication decision.

## Write a literature report

Give the drafter the selected question and the saved results. After the
editor checks the summary, conclusion and proposed future work using the
three reader questions in the drafting and editing roles, the study owner
applies any repairs and approves the copy against the paper and run evidence.

Give the reader this sequence. Choose the published headings from the science of the particular study.

1. Pair the table-of-contents (TOC) figure with an unheaded 100–200 word opening directly below the title. Let the figure show the comparison and the prose explain the question, the reference study, the principal result and its implication. Include the decisive numbers and conditions. When a report has no TOC figure, the opening still stands on its own.
2. Give the paper's background a descriptive heading naming the experiment, phenomenon or question that motivated it. In fewer than 300 words, explain what the authors measured or modeled and how their work motivates the comparison.
3. Present the repeated comparison before the Particula extension. Label measured series, the authors' model and each Particula case in the figures. Introduce follow-ups with the gap or discrepancy that motivated them. In a large study, keep the main report readable on its own and give each supporting page a physical question, result and replay. Link it where its finding changes the main interpretation.
4. Close the scientific narrative under a heading that states the takeaway or calls for a specific next test. Explain what the result lets the reader infer, where that inference is limited and what evidence could change it.
5. End with reproducibility information and study resources. Collect inputs, run conditions, versions, commands and saved outputs there. A **Study resources** section can serve this purpose on its own or follow **Reproducibility** with the site's generated links. Place footnote definitions at the end and keep reference markers beside the material they credit.

Let the reader see the contribution through a specific consequence, such as an assumption that reverses a prediction or an archived channel that distinguishes two explanations. Credit the original study and identify the model where needed for attribution.

Write the headline as the answer you would give a researcher. Lead with the comparison that has the clearest scientific consequence and name the observable, direction and decisive condition. A longer title earns its length when the added detail changes the interpretation. For example, “Particula predicts half the rise in ion concentration CLOUD measured from 7% to 70% humidity.” The counts establish a concentration discrepancy. Its cause could involve wall loss, production or sampling. When several explanations fit, the headline can say what the data constrain or which test would distinguish them.

When revisiting a report with several findings, ask which comparison would most change a reader’s interpretation or next experiment. A later test can earn the lead if its controls and uncertainty support that inference. State the observable, time, reference case and regime that give the effect its meaning. Check whether the ranking holds in another tested regime. A large percentage alone is a poor reason to promote a result.

Use section headings to carry that argument from the evidence to its limits and the next test. For an approved runnable example in the Particula repository, use `../particula-examples/SKILL.md`.

After the narrative revision, read the final title, opening and conclusion beside the existing TOC figure. If the lead finding, comparator or interpretation changed, revise the figure and its alt text wherever the collection displays it. Keep artwork that still conveys the final comparison. For a new or revised graphical abstract, use `../particula-toc/SKILL.md` with the final title and comparison.

## Shared operating rules
Use the selected interpreter’s `python -m particula_docs --json` to locate and search version-matched installed docs before relying on the live site; see `../../references/sources.md` for commands and older-version fallback.
Read `../../references/compatibility.md` for tested versions and known numerical limitations.
Read `../../references/scientific-practice.md` before technical work and use `../../references/sources.md` to find authoritative documentation. Resolve these paths relative to this skill file, not the user's working directory.
Use the user's chosen local project and Python environment. Treat bundled files as read-only templates; write studies and results to the user's project. A host with local execution and file access is required to run on the user's computer. If unavailable, explain the limitation and provide runnable files; never describe proposed code as executed.
Check the installed version and actual signatures before using an API. Do not silently upgrade, replace a user's environment, or switch to the development branch to make an example work. Keep progress moving with reasonable, explicit scientific defaults for demonstrations; ask only for missing inputs that materially affect the requested study.
