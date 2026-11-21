---
name: particula-studies-literature
description: "Analyze Particula simulations and write literature studies comparing model predictions with published measurements or theory. Use for quantitative figures, conservation, convergence, uncertainty and finding-led research reports."
---

# Analyze aerosol studies and write literature reports

Use the [drafting and editing roles](../../references/drafting-and-editing.md)
when framing a study question or writing its report. An independently launched
literature analysis can originate the question. The study owner can select
it from the primary evidence and proceed without an intake interview.

## Analyze the results

1. Read run metadata and numerical outputs before interpreting plots. Establish units, distribution representation, radius/diameter convention, species and process configuration.
2. Check NaN/Inf, negative concentrations/masses, number and mass trends, and the correct open/closed-system inventory. Condensation exchanges gas/particle mass; coagulation changes number while conserving mass in the ideal closed model. Boundary losses or sources require explicit budgets.
3. Distinguish per-bin counts, PDFs and dN/dlogD; integrate with appropriate widths and Jacobians. Never sum a continuous density as though it were per-bin counts.
4. Compare smaller time steps and finer resolution for numerical convergence. Report changes in physically relevant observables, not only plot appearance. For stochastic studies use independent seeds and uncertainty intervals when available.
5. Label axes and units; use logarithmic size axes where appropriate. Preserve numerical observations and export plots plus a short report of inputs, versions, checks and limitations. Reuse saved data for plotting-only changes; do not rerun physics or reconstruct values from images. Mark sampled observations explicitly rather than implying exact event times. Registered repository examples follow `../particula-docs/SKILL.md` for stable folders, bounded plot data and replot provenance.
6. For observations align units, sampling windows, instrument response and uncertainties; do not conflate calibration and independent validation. Separate numerical, parameter and structural uncertainty.
Deliver key findings, linked plots/data, quantitative check results and justified next steps. A successful bundled smoke test is not evidence that a model predicts a particular experiment.

## Write a literature report

Give the drafter the selected question and the saved results. After the
editor checks the summary, conclusion and proposed future work using the
three reader questions in the drafting and editing roles, the study owner
applies any repairs and approves the copy against the paper and run evidence.

Give the reader this sequence. Choose the published headings from the science of the particular study.

1. Pair the table-of-contents (TOC) figure with an unheaded 100–200 word opening directly below the title. Let the figure show the comparison and the prose explain the question, the reference study, the principal result and its implication. Include the decisive numbers and conditions. When a report has no TOC figure, the opening still stands on its own.
2. Give the paper's background a descriptive heading naming the experiment, phenomenon or question that motivated it. In fewer than 300 words, explain what the authors measured or modeled and how their work motivates the comparison.
3. Introduce the Particula analysis under a heading that names the scientific test or finding. Nest the individual comparisons and extensions beneath it as subsections with informative headings. Distinguish observations, fitted quantities, predictions and proposed tests. Link the specific figure, table or equation beside each comparison.
4. Close the scientific narrative under a heading that states the takeaway or calls for a specific next test. Explain what the result lets the reader infer, where that inference is limited and what evidence could change it.
5. End with reproducibility information and study resources. Collect inputs, run conditions, versions, commands and saved outputs there. A **Study resources** section can serve this purpose on its own or follow **Reproducibility** with the site's generated links. Place footnote definitions at the end and keep reference markers beside the material they credit.

Let the reader see the contribution through a specific consequence, such as an assumption that reverses a prediction or a measurement that could distinguish two explanations. Credit the original study and identify the model where needed for attribution.

Write the headline as the answer you would give a researcher. Lead with a surprising, defensible comparison and name the observable, direction and decisive condition. A longer title earns its length when the added detail changes the interpretation. For example, “Particula predicts half the rise in ion concentration CLOUD measured from 7% to 70% humidity.” The counts establish a concentration discrepancy. Its cause could involve wall loss, production or sampling. When several explanations fit, the headline can say what the data constrain or which test would distinguish them.

Use section headings to carry that argument from the evidence to its limits and the next test. For an approved runnable example in the Particula repository, use `../particula-examples/SKILL.md`.

When a study needs a graphical abstract for its contents listing, use `../particula-toc/SKILL.md` with the final title and comparison.

## Shared operating rules
Use the selected interpreter’s `python -m particula_docs --json` to locate and search version-matched installed docs before relying on the live site; see `../../references/sources.md` for commands and older-version fallback.
Read `../../references/compatibility.md` for tested versions and known numerical limitations.
Read `../../references/scientific-practice.md` before technical work and use `../../references/sources.md` to find authoritative documentation. Resolve these paths relative to this skill file, not the user's working directory.
Use the user's chosen local project and Python environment. Treat bundled files as read-only templates; write studies and results to the user's project. A host with local execution and file access is required to run on the user's computer. If unavailable, explain the limitation and provide runnable files; never describe proposed code as executed.
Check the installed version and actual signatures before using an API. Do not silently upgrade, replace a user's environment, or switch to the development branch to make an example work. Keep progress moving with reasonable, explicit scientific defaults for demonstrations; ask only for missing inputs that materially affect the requested study.
