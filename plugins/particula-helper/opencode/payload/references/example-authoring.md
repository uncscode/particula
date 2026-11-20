# Adaptable format for published examples

Use this structure when creating or editing a worked example. Rename, combine
or repeat sections to suit the scientific problem; keep the narrative purposes.
A focused edit need not restructure the whole page. Preserve its scope and
record remaining editorial gaps rather than silently expanding the assignment.

## Default page structure

| Stage | What the reader needs |
| --- | --- |
| Question and motivation | A descriptive title, the scientific question, why it matters and what the reader will be able to interpret or do. |
| Physical setup | Assumptions, initial/boundary conditions, important parameter choices and units, prerequisites and backend requirements. Add a schematic when it helps explain the system. |
| Prepare the calculation | Explain imports, configuration, data representation and initial state before the corresponding code blocks. Link unfamiliar concepts rather than repeating whole guides. |
| Run and check | Explain each operation before its code; show the diagnostic or independent check that establishes what happened. State process order, duration, time steps and seeds where relevant. |
| Visualize and interpret | Plot the relevant results or show a meaningful process visual. Explain what to look for and what the displayed evidence supports, near the figure. |
| Conclusions and reflections | Answer the opening question, explain mechanisms and limitations, distinguish numerical checks from physical validation, and identify what remains uncertain. |
| Next investigation and connections | Propose a concrete change, comparison and diagnostic; identify any extra evidence needed. Link prerequisites, relevant theory and a useful next example. Mark unrun work as proposed. |

Use ordinary prose in `# %% [markdown]` cells around executable `# %%` cells.
Every code block needs a nearby introduction explaining why it is there,
important inputs and units, and the expected output or diagnostic. Include
imports/setup and helper-definition cells; group small related fragments when
they share an explanation. Comments inside Python supplement this prose rather
than replacing it. Avoid narrating obvious syntax line by line.

When a helper produces its figure only at a later invocation, put the result's
interpretation beside that invocation/output. Do not move prose across cells
without checking the rendered output association. Preserve the source's
registered invocation and avoid accidentally running the workload twice.

## Visuals as part of the explanation

Plan at least one useful visual by default. Most quantitative examples need a
results plot generated from their computed or verified saved data. Add a
problem/process visual when geometry, flows, competing mechanisms, population
measures or execution stages would otherwise be hard to follow. A qualitative
API/lifecycle example may use a schematic as its main visual; a numerical claim
still needs its supporting computed evidence. If neither kind of visual would
help a narrow example, explain that editorial exception in the delivery notes.

- Choose a plot that answers the question: evolution, before/after distributions,
  a comparison with an independent reference, a sensitivity or an error/conservation
  diagnostic. Do not add a decorative chart merely to meet the default.
- Generate quantitative figures with the example's plotting code. Label axes,
  units, measures (including radius versus diameter), scales, cases and reference
  provenance. Distinguish simulation, theory and observations. Show uncertainty
  when the example's claims require it; do not manufacture confidence intervals.
- Use a legible diagram for conceptual structure. Prefer repository-supported
  local vector assets or reproducible drawing code for simple schematics. Include
  meaningful labels, caption and text description/alt text where supported.
  Explicitly identify schematic geometry or illustrative artwork as such.
- Make legends, labels and captions readable at the rendered page size. Use
  contrast and line styles/markers so color alone does not carry the distinction.
  Explain the important trend and evidence limits in adjacent prose. Save plots
  at the repository's publication resolution and inspect the actual images.

Store static authored visuals in the owning docs image location with relative
links; store generated results through the example result workflow. A topic
card's icon provides navigation context and does not replace an example visual.

## Registration and topic completion

Before calling a new example complete, verify all of these in the selected
checkout, using the current schemas and a comparable neighboring example:

- `website/examples.json` registers its source, invocation, dependencies,
  execution requirements and saved-data rendering contract where applicable.
- `website/topics.json` associates the source with one primary scientific topic,
  appropriate additional topics and its role. Reconsider `start_here` placement;
  do not make every new example a starting resource.
- Its main topic resource owns one study record and primary browsing home
  under `docs/studies/`. Register run keys, companion material, and specific
  guide, equation, evidence or API relationships there. Check the generated
  category landing, code index and backlinks. Preserve meaningful old routes
  and fragments with the relocation table.
- Topic prerequisites and next-reading links lead to useful destinations. New
  concepts follow the branching, relationship and icon rules in
  [documentation-topics.md](documentation-topics.md); do not invent categories
  just to house one example when an existing concept fits.

Run `python scripts/docs_topics.py`, `python scripts/check_docs_structure.py`,
the applicable example validation/tests, and `npm --prefix website run build`.
Then check the rendered page and topic links. Refresh simulation results for
scientific changes, replot for plot-only changes, and retain valid results for
prose-only edits, following the checkout's `website/AUTHORING.md`.
